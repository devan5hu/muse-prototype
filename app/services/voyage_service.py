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
        img.thumbnail((256, 256))  # Resize to reasonable dimensions
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
            model="voyage-multimodal-3",
            input_type="query"
        )
        embedding = result.embeddings[0]
        print(f"inputs: {inputs[0]}")
        print(f"Embedding shape: {embedding}")
        
        if result_queue:
            result_queue.put(("success", embedding))
        return embedding
    except Exception as e:
        print(f"Voyage API error details: {str(e)}")
        if result_queue:
            result_queue.put(("error", str(e)))
        raise e

def get_voyage_embedding(text=None, img=None, max_retries=3, request_timeout=15):
    """Generate embedding with a manual timeout and retry."""
    # At least one of text or image must be provided
    if not text and not img:
        raise ValueError("At least one of text or image_path must be provided")
    
    # Ensure text is a string
    if text is None:
        text = ""
    
    for attempt in range(max_retries):
        print(f"Attempting Voyage API call (Attempt {attempt + 1}/{max_retries})...")
        print(f"Input: text='{text}' (type: {type(text)}), image={type(img)}")
        
        result_queue = queue.Queue()
        thread = threading.Thread(target=run_embedding_request, args=(text, img, result_queue))
        
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