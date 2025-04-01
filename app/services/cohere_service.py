import os
import io
import base64
import queue
import threading
from time import time, sleep
import numpy as np
from PIL import Image
import cohere
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Cohere client
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(COHERE_API_KEY)

def image_to_base64(file_path):
    """Convert image to base64 data URI, handling PNG palette mode."""
    with open(file_path, 'rb') as image_file:
        img = Image.open(image_file)
        if img.mode == 'P':  # Convert palette mode to RGB
            img = img.convert('RGB')
        img.thumbnail((256, 256))  # Resize to 256x256
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"  # Return as data URI

def run_embedding_request(images=None, texts=None, result_queue=None):
    """Run the Cohere embedding request and store the result in a queue."""
    try:
        response = co.embed(
            images=images if images else None,
            texts=texts if texts else None,
            model="embed-english-v3.0",
            input_type="image" if images else "search_query",
            embedding_types=["float"]
        )
        embedding = response.embeddings.float_[0]
        if result_queue:
            result_queue.put(("success", embedding))
        return embedding
    except Exception as e:
        if result_queue:
            result_queue.put(("error", str(e)))
        raise e

def get_cohere_embedding(image_path, max_retries=3, request_timeout=10):
    """Generate embedding with a manual 10-second timeout and retry."""
    base64_image = image_to_base64(image_path)
    print(f"Converted {image_path} to base64 (length: {len(base64_image)} characters)")
    
    for attempt in range(max_retries):
        print(f"Attempting API call (Attempt {attempt + 1}/{max_retries})...")
        result_queue = queue.Queue()
        thread = threading.Thread(target=run_embedding_request, args=([base64_image], None, result_queue))
        
        thread.start()
        thread.join(timeout=request_timeout)  # Wait up to 10 seconds
        
        if thread.is_alive():
            # If thread is still running after 10 seconds, assume timeout
            print(f"Request took longer than {request_timeout} seconds.")
            if attempt < max_retries - 1:
                print(f"Retrying in 10 seconds... (Attempt {attempt + 1}/{max_retries})")
                continue
            else:
                raise Exception(f"Failed after {max_retries} retries: Request exceeded {request_timeout} seconds")
        
        # Thread finished within 10 seconds, get the result
        status, result = result_queue.get()
        if status == "success":
            print("API call succeeded.")
            return result
        else:
            print(f"API error: {result}")
            if "rate limit" in str(result).lower() or "429" in str(result):
                if attempt < max_retries - 1:
                    wait_time = 5 * (2 ** attempt)
                    print(f"Rate Limit Error: Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    sleep(wait_time)
                else:
                    raise Exception(f"Failed after {max_retries} retries due to rate limit: {result}")
            else:
                raise Exception(f"API failed: {result}")

def get_text_embedding(text, max_retries=3, request_timeout=10):
    """Generate text embedding with timeout and retry."""
    for attempt in range(max_retries):
        print(f"Attempting text embedding API call (Attempt {attempt + 1}/{max_retries})...")
        result_queue = queue.Queue()
        thread = threading.Thread(target=run_embedding_request, args=(None, [text], result_queue))
        
        thread.start()
        thread.join(timeout=request_timeout)
        
        if thread.is_alive():
            print(f"Request took longer than {request_timeout} seconds.")
            if attempt < max_retries - 1:
                print(f"Retrying in 10 seconds...")
                sleep(1)
                continue
            else:
                raise Exception(f"Failed after {max_retries} retries: Request exceeded {request_timeout} seconds")
        
        status, result = result_queue.get()
        if status == "success":
            print("Text embedding API call succeeded.")
            return result
        else:
            print(f"API error: {result}")
            if "rate limit" in str(result).lower() or "429" in str(result):
                if attempt < max_retries - 1:
                    wait_time = 5 * (2 ** attempt)
                    print(f"Rate Limit Error: Retrying in {wait_time} seconds...")
                    sleep(wait_time)
                else:
                    raise Exception(f"Failed after {max_retries} retries due to rate limit: {result}")
            else:
                raise Exception(f"API failed: {result}")

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

def search_images(query_text, stored_embeddings, image_paths=None, top_k=7, query_image_path=None, image_weight=0.4):
    """
    Search for images using both text and image embeddings with weighted similarity.
    
    Args:
        query_text: Text description to search for
        stored_embeddings: Dictionary or list of stored embeddings
        image_paths: List of image paths corresponding to the embeddings
        top_k: Number of top results to return
        query_image_path: Path to query image (optional)
        image_weight: Weight for image similarity (0-1), text weight will be (1-image_weight)
        
    Returns:
        List of tuples (image_path, combined_similarity, image_similarity, text_similarity)
    """
    # If image_paths is not provided, extract from stored_embeddings
    if image_paths is None and isinstance(stored_embeddings, dict) and "image_paths" in stored_embeddings:
        image_paths = stored_embeddings["image_paths"]
        stored_embeddings = stored_embeddings["embeddings"]
    
    # Generate text embedding for the query
    print(f"Generating text embedding for query: {query_text[:50]}...")
    query_text_embedding = get_text_embedding(query_text)
    
    # Generate image embedding for the query if image path is provided
    query_image_embedding = None
    if query_image_path and os.path.exists(query_image_path):
        print(f"Generating image embedding for query image: {query_image_path}")
        query_image_embedding = get_cohere_embedding(query_image_path)
    
    # Compute similarities
    results = []
    for idx, stored_emb in enumerate(stored_embeddings):
        if stored_emb is None:  # Skip failed embeddings
            continue
            
        # Get the corresponding image path
        img_path = image_paths[idx] if image_paths and idx < len(image_paths) else f"image_{idx}"
        
        # Compute text similarity
        text_similarity = cosine_similarity(query_text_embedding, stored_emb)
        
        # Compute image similarity if query image is provided
        image_similarity = None
        if query_image_embedding is not None:
            image_similarity = cosine_similarity(query_image_embedding, stored_emb)
            
            # Compute combined similarity with weighting
            combined_similarity = (image_weight * image_similarity) + ((1 - image_weight) * text_similarity)
        else:
            # If no image query, use only text similarity
            combined_similarity = text_similarity
        
        results.append((img_path, combined_similarity, image_similarity, text_similarity))
    
    # Sort by combined similarity (descending) and get top_k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k] 