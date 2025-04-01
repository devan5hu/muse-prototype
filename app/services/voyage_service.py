import os
import io
import base64
import queue
import threading
from time import time, sleep
import numpy as np
from PIL import Image
import voyageai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Voyage AI client
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
client = voyageai.Client(api_key=VOYAGE_API_KEY)

def image_to_pil(file_path):
    """Convert image file path to PIL Image."""
    try:
        img = Image.open(file_path)
        if img.mode == 'P':  # Convert palette mode to RGB
            img = img.convert('RGB')
        img.thumbnail((512, 512))  # Resize to reasonable dimensions
        return img
    except Exception as e:
        print(f"Error loading image {file_path}: {str(e)}")
        raise e

def run_embedding_request(text=None, image=None, result_queue=None):
    """Run the Voyage embedding request and store the result in a queue."""
    try:
        # The Voyage API expects different formats depending on what's provided
        if image is not None and text:
            # Both text and image
            inputs = [[text, image]]
        elif image is not None:
            # Only image
            inputs = [[image]]
        elif text:
            # Only text
            inputs = [[text]]
        else:
            # Neither - shouldn't happen but handle it
            raise ValueError("Either text or image must be provided")
        
        print(f"Input types: text={type(text) if text else None}, image={type(image) if image else None}")
        
        result = client.multimodal_embed(
            inputs=inputs,
            model="voyage-multimodal-3"
        )
        embedding = result.embeddings[0]
        
        if result_queue:
            result_queue.put(("success", embedding))
        return embedding
    except Exception as e:
        print(f"Voyage API error details: {str(e)}")
        if result_queue:
            result_queue.put(("error", str(e)))
        raise e

def get_voyage_embedding(text=None, image_path=None, max_retries=3, request_timeout=15):
    """Generate embedding with a manual timeout and retry."""
    # At least one of text or image must be provided
    if not text and not image_path:
        raise ValueError("At least one of text or image_path must be provided")
    
    # Ensure text is a string
    if text is None:
        text = ""
    
    # Load image if path is provided
    image = None
    if image_path:
        print(f"Loading image from {image_path}")
        try:
            image = image_to_pil(image_path)
            print(f"Image loaded successfully: {type(image)}, mode: {image.mode}, size: {image.size}")
        except Exception as e:
            print(f"Error loading image: {e}")
            # If image loading fails but we have text, continue with just text
            if text:
                print("Continuing with text only since image loading failed")
            else:
                raise ValueError("Image loading failed and no text provided")
    
    for attempt in range(max_retries):
        print(f"Attempting Voyage API call (Attempt {attempt + 1}/{max_retries})...")
        print(f"Input: text='{text}' (type: {type(text)}), image={type(image)}")
        
        result_queue = queue.Queue()
        thread = threading.Thread(target=run_embedding_request, args=(text, image, result_queue))
        
        thread.start()
        thread.join(timeout=request_timeout)  # Wait up to 15 seconds
        
        if thread.is_alive():
            # If thread is still running after timeout, assume timeout
            print(f"Request took longer than {request_timeout} seconds.")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds... (Attempt {attempt + 1}/{max_retries})")
                sleep(5)
                continue
            else:
                raise Exception(f"Failed after {max_retries} retries: Request exceeded {request_timeout} seconds")
        
        # Thread finished within timeout, get the result
        status, result = result_queue.get()
        if status == "success":
            print("Voyage API call succeeded.")
            return result
        else:
            print(f"Voyage API error: {result}")
            if "rate limit" in str(result).lower() or "429" in str(result):
                if attempt < max_retries - 1:
                    wait_time = 5 * (2 ** attempt)
                    print(f"Rate Limit Error: Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    sleep(wait_time)
                else:
                    raise Exception(f"Failed after {max_retries} retries due to rate limit: {result}")
            else:
                raise Exception(f"Voyage API failed: {result}")

def cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    # Convert to numpy arrays if they aren't already
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    
    # Compute dot product and norms
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    # Compute cosine similarity
    similarity = dot_product / (norm1 * norm2)
    return similarity 