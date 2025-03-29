import os
import logging
import json
import base64
import csv
import tempfile
from io import BytesIO
from PIL import Image
import boto3
import requests

logger = logging.getLogger(__name__)

# Configuration
AWS_REGION = "ap-southeast-2"  # Titan Multimodal is available here
CSV_KEY_PATH = "static/devan5hu-bedrock-user_accessKeys.csv"

# Initialize client at module level
bedrock_client = None

def resize_image(image_path, max_size=(1024, 1024)):
    """
    Resize image if needed and return as bytes
    
    Args:
        image_path: Path to the image file
        max_size: Maximum dimensions (width, height)
        
    Returns:
        bytes: Image data as bytes
    """
    try:
        with Image.open(image_path) as img:
            # Only resize if the image is larger than max_size
            if img.width > max_size[0] or img.height > max_size[1]:
                img.thumbnail(max_size, Image.LANCZOS)
                
            # Convert to RGB if needed (e.g., for PNG with transparency)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Save to bytes
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            return buffer.getvalue()
            
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        raise

def load_aws_credentials(csv_path):
    """
    Load AWS credentials from CSV file
    
    Args:
        csv_path: Path to the CSV file containing AWS credentials
        
    Returns:
        tuple: (access_key_id, secret_access_key)
    """
    try:
        with open(csv_path, "r", encoding='utf-8-sig') as f:  # utf-8-sig handles the BOM
            reader = csv.DictReader(f)
            for row in reader:
                # Use the correct key with BOM
                key_id = row.get('\ufeffAccess key ID') or row.get('Access key ID')
                secret_key = row['Secret access key']
                return key_id, secret_key
    except Exception as e:
        logger.error(f"Error reading AWS credentials CSV: {str(e)}")
        raise

def initialize_bedrock_client():
    """Initialize the AWS Bedrock client"""
    global bedrock_client
    
    try:
        # Check internet connectivity first
        try:
            requests.get('https://www.google.com', timeout=5)
        except requests.exceptions.RequestException:
            logger.error("No internet connection available")
            return None
            
        # Try loading credentials
        try:
            access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
            secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
            logger.info(f"AWS credentials loaded successfully: Access Key ID: {access_key_id[:5]}...")
        except Exception as e:
            logger.error(f"Failed to load AWS credentials: {str(e)}")
            return None

        # Initialize Bedrock runtime client with CSV credentials
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
        
        # Also create a regular bedrock client to list available models
        bedrock_mgmt = boto3.client(
            service_name="bedrock",
            region_name=AWS_REGION,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
        
        try:
            # List available foundation models to verify access
            models = bedrock_mgmt.list_foundation_models()
            logger.info(f"Available Bedrock models: {[m['modelId'] for m in models.get('modelSummaries', [])]}")
            
            # Check if our required models are available
            available_models = [m['modelId'] for m in models.get('modelSummaries', [])]
            required_models = ["amazon.titan-embed-image-v1"]
            
            for model in required_models:
                if model not in available_models:
                    logger.warning(f"Required model {model} is not available in your AWS account/region")
        except Exception as e:
            logger.warning(f"Could not list available models: {str(e)}")
        
        logger.info("AWS Bedrock client initialized successfully")
        return client
        
    except Exception as e:
        logger.error(f"Error initializing AWS Bedrock client: {str(e)}", exc_info=True)
        return None

# Initialize the client
bedrock_client = initialize_bedrock_client()

def get_titan_embedding(text=None, image_path=None):
    """
    Generate embeddings using Titan Multimodal Embeddings model
    
    This unified function handles text-only, image-only, or multimodal (text+image) embeddings
    
    Args:
        text: Optional text to generate embeddings for
        image_path: Optional path to image file
        
    Returns:
        dict: Embedding results
    """
    try:
        global bedrock_client
        
        # Validate inputs
        if text is None and image_path is None:
            logger.error("Both text and image_path cannot be None")
            raise ValueError("Either text or image_path (or both) must be provided")
        
        # If client is not initialized, try to initialize it again
        if bedrock_client is None:
            logger.info("Bedrock client not initialized, attempting to initialize now")
            bedrock_client = initialize_bedrock_client()
            
        if bedrock_client is None:
            logger.error("Could not initialize AWS Bedrock client")
            raise RuntimeError("AWS Bedrock service unavailable")
            
        # Determine embedding type for logging
        if text and image_path:
            embedding_type = "multimodal"
            logger.info(f"Processing with Titan {embedding_type}: image={image_path}, text={text[:50] if text else ''}...")
        elif image_path:
            embedding_type = "image"
            logger.info(f"Processing with Titan {embedding_type}: image={image_path}")
        else:
            embedding_type = "text"
            logger.info(f"Processing with Titan {embedding_type}: text={text[:50] if text else ''}...")
        
        # Prepare request body
        body = {
            "embeddingConfig": {"outputEmbeddingLength": 256}
        }
        
        # Add text if provided
        if text:
            body["inputText"] = text
            
        # Add image if provided
        if image_path:
            # Resize if necessary and encode to base64
            image_data = resize_image(image_path)
            encoded_image = base64.b64encode(image_data).decode("utf-8")
            body["inputImage"] = encoded_image
        
        # Invoke Titan Multimodal Embeddings model
        response = bedrock_client.invoke_model(
            body=json.dumps(body),
            modelId="amazon.titan-embed-image-v1",
            accept="application/json",
            contentType="application/json"
        )
        
        # Parse response
        result = json.loads(response["body"].read())
        
        logger.info(f"Successfully obtained {embedding_type} embeddings from Titan")
        return {
            "embedding": result["embedding"],
            "embedding_type": embedding_type
        }
        
    except Exception as e:
        logger.error(f"Error getting Titan embeddings: {str(e)}", exc_info=True)
        raise

def get_titan_image_embedding(image_path):
    """Wrapper for backward compatibility"""
    try:
        return get_titan_embedding(image_path=image_path)
    except Exception as e:
        logger.error(f"Error in image embedding: {str(e)}")
        raise

def get_titan_text_embedding(text):
    """Wrapper for backward compatibility"""
    try:
        return get_titan_embedding(text=text)
    except Exception as e:
        logger.error(f"Error in text embedding: {str(e)}")
        raise

def get_titan_multimodal_embedding(image_path, text):
    """Wrapper for backward compatibility"""
    try:
        return get_titan_embedding(text=text, image_path=image_path)
    except Exception as e:
        logger.error(f"Error in multimodal embedding: {str(e)}")
        raise