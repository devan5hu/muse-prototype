import os
import logging
import json
import tempfile
import time
import requests
import socket
from io import BytesIO
import PIL
from PIL import Image
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Image as VertexImage
from google.api_core import retry
import dotenv
from os import environ

logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

# Vertex AI configuration
VERTEX_PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID")
VERTEX_LOCATION = "us-central1"
VERTEX_ENDPOINT = "us-central1-aiplatform.googleapis.com"

# Service account credentials from environment variables
SERVICE_ACCOUNT_INFO = {
    "type": "service_account",
    "project_id": os.environ.get("GOOGLE_PROJECT_ID"),
    "private_key_id": os.environ.get("GOOGLE_PRIVATE_KEY_ID"),
    "private_key": os.environ.get("GOOGLE_PRIVATE_KEY"),
    "client_email": os.environ.get("GOOGLE_CLIENT_EMAIL"),
    "client_id": os.environ.get("GOOGLE_CLIENT_ID"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": os.environ.get("GOOGLE_CLIENT_X509_CERT_URL"),
    "universe_domain": os.environ.get("GOOGLE_UNIVERSE_DOMAIN", "googleapis.com")
}

def check_dns_resolution(hostname):
    """Check if DNS resolution works for a specific hostname"""
    try:
        socket.gethostbyname(hostname)
        return True
    except socket.gaierror:
        return False

# Initialize the model at module level
def initialize_vertex_ai():
    """Initialize Vertex AI and load the model"""
    global model
    
    try:
        # Check internet connectivity first
        try:
            # Try to connect to Google to verify internet connectivity
            requests.get('https://www.google.com', timeout=5)
        except requests.exceptions.RequestException:
            logger.error("No internet connection available")
            return None
        
        # Check DNS resolution for Vertex AI endpoint
        if not check_dns_resolution(VERTEX_ENDPOINT):
            logger.error(f"DNS resolution failed for {VERTEX_ENDPOINT}")
            logger.info("Attempting to use alternative DNS servers...")
            
            # Try to use Google's public DNS servers
            # This is a suggestion for environments where the default DNS might be failing
            try:
                # This is a simple test to see if Google's DNS can resolve the hostname
                response = requests.get(f"https://dns.google/resolve?name={VERTEX_ENDPOINT}", timeout=5)
                if response.status_code == 200:
                    logger.info("Google DNS lookup successful, but your system DNS is still not working properly")
            except Exception as e:
                logger.warning(f"Google DNS test failed: {str(e)}")
            
            return None
            
        # Create a temporary file to store the credentials
        temp_key_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        temp_key_path = temp_key_file.name
        
        # Write the credentials to the temporary file
        with open(temp_key_path, 'w') as f:
            json.dump(SERVICE_ACCOUNT_INFO, f)
        
        # Set the environment variable to point to the temporary file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_key_path
        
        # Initialize Vertex AI with retry
        for attempt in range(3):
            try:
                vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
                logger.info(f"Vertex AI initialized for project: {VERTEX_PROJECT_ID}, region: {VERTEX_LOCATION}")
                
                # Load the multimodal embedding model
                model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
                logger.info("Multimodal embedding model loaded successfully")
                
                return model
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
                if "DNS resolution failed" in str(e):
                    logger.error("DNS resolution issue detected. Please check your network configuration.")
                    # Breaking early on DNS issues as retries are unlikely to help
                    break
                if attempt < 2:  # Don't sleep on the last attempt
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error("All initialization attempts failed")
        return None
    except Exception as e:
        logger.error(f"Error initializing Vertex AI: {str(e)}", exc_info=True)
        return None

# Initialize the model
model = initialize_vertex_ai()

def get_vertex_embeddings(image_path, text):
    """
    Get embeddings from Vertex AI multimodal model
    
    Args:
        image_path: Path to the image file
        text: Text to process alongside the image
        
    Returns:
        dict: Embedding results
    """
    try:
        global model
        
        # If model is not initialized, try to initialize it again
        if model is None:
            logger.info("Model not initialized, attempting to initialize now")
            model = initialize_vertex_ai()
            
        if model is None:
            # If still None, return a fallback response
            logger.error("Could not initialize Vertex AI model - returning fallback response")
            return {
                "error": "Service unavailable - DNS resolution failed for Vertex AI endpoint",
                "text_embedding": [0.0] * 256,  # Dummy embeddings
                "image_embedding": [0.0] * 256,
                "multimodal_embedding": [0.0] * 256
            }
            
        logger.info(f"Processing with Vertex AI: image={image_path}, text={text}")
        
        # Check DNS resolution before attempting to use the model
        if not check_dns_resolution(VERTEX_ENDPOINT):
            logger.error(f"DNS resolution failed for {VERTEX_ENDPOINT}")
            return {
                "error": "Service unavailable - DNS resolution failed for Vertex AI endpoint",
                "text_embedding": [0.0] * 256,  # Dummy embeddings
                "image_embedding": [0.0] * 256,
                "multimodal_embedding": [0.0] * 256
            }
        
        # Load image using vertexai's Image class, not PIL
        image = VertexImage.load_from_file(image_path)
        
        # Get embeddings with retry
        for attempt in range(3):
            try:
                embeddings = model.get_embeddings(
                    image=image,
                    contextual_text=text,
                    dimension=256
                )
                
                # Convert embeddings to serializable format
                result = {
                    "text_embedding": embeddings.text_embedding.values.tolist() if embeddings.text_embedding else None,
                    "image_embedding": embeddings.image_embedding.values.tolist() if embeddings.image_embedding else None,
                    "multimodal_embedding": embeddings.multimodal_embedding.values.tolist() if embeddings.multimodal_embedding else None,
                }
                
                logger.info("Successfully obtained embeddings from Vertex AI")
                return result
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
                if "DNS resolution failed" in str(e):
                    logger.error("DNS resolution issue detected during embedding generation.")
                    # Breaking early on DNS issues as retries are unlikely to help
                    break
                if attempt < 2:  # Don't sleep on the last attempt
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # If all attempts fail, return a fallback response
        logger.error("All embedding attempts failed")
        return {
            "error": "Service unavailable after multiple attempts - DNS resolution issues",
            "text_embedding": [0.0] * 256,  # Dummy embeddings
            "image_embedding": [0.0] * 256,
            "multimodal_embedding": [0.0] * 256
        }
        
    except Exception as e:
        logger.error(f"Error getting Vertex AI embeddings: {str(e)}", exc_info=True)
        # Return fallback response instead of raising
        return {
            "error": str(e),
            "text_embedding": [0.0] * 256,  # Dummy embeddings
            "image_embedding": [0.0] * 256,
            "multimodal_embedding": [0.0] * 256
        } 