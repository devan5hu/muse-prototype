import os
import logging
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# Configuration
EMBEDDINGS_JSON_PATH = "static/json/titan.json"

def load_embeddings():
    """
    Load embeddings from JSON file
    
    Returns:
        dict: Dictionary with embeddings and file paths
    """
    try:
        if not os.path.exists(EMBEDDINGS_JSON_PATH):
            logger.error(f"Embeddings file not found: {EMBEDDINGS_JSON_PATH}")
            return None
            
        with open(EMBEDDINGS_JSON_PATH, 'r') as f:
            embeddings_data = json.load(f)
            
        logger.info(f"Loaded embeddings from JSON file: {len(embeddings_data)} entries")
        return embeddings_data
        
    except Exception as e:
        logger.error(f"Error loading embeddings from JSON: {str(e)}", exc_info=True)
        return None

def get_image_url(file_path):
    """Convert a file path to a URL that can be accessed from the frontend"""
    # Extract just the filename if it's a full path
    filename = os.path.basename(file_path)
    
    # Simply return the path as a static URL
    # This assumes the images are directly in the static folder or a subfolder
    return f"/static/all_images/{filename}"

def find_similar_images(query_embedding, top_n=10):
    """
    Find similar images based on cosine similarity
    
    Args:
        query_embedding: Embedding vector to compare against
        top_n: Number of top results to return
        
    Returns:
        list: List of dictionaries with similarity scores and file paths
    """
    try:
        # Load embeddings data
        embeddings_data = load_embeddings()
        if not embeddings_data:
            logger.error("Could not load embeddings data")
            return []
            
        # Extract embeddings and file paths
        stored_embeddings = []
        file_paths = []
        
        # Debug the structure of the embeddings data
        logger.info(f"Embeddings data type: {type(embeddings_data)}")
        
        # Handle JSON structure - expect a dictionary with image IDs as keys
        for image_id, data in embeddings_data.items():
            if isinstance(data, dict) and 'embedding' in data:
                file_paths.append(data.get('path', image_id))
                stored_embeddings.append(data['embedding'])
            else:
                logger.warning(f"Skipping entry with unexpected format: {image_id}")
        
        logger.info(f"Extracted {len(file_paths)} file paths and {len(stored_embeddings)} embeddings")
        
        if not stored_embeddings or not file_paths:
            logger.error("Could not extract embeddings and file paths from the loaded data")
            return []
            
        # Convert to numpy arrays for efficient computation
        try:
            stored_embeddings_array = np.array(stored_embeddings, dtype=np.float32)
            query_embedding_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            
            logger.info(f"Stored embeddings shape: {stored_embeddings_array.shape}")
            logger.info(f"Query embedding shape: {query_embedding_array.shape}")
            
            first_norm = np.linalg.norm(stored_embeddings_array[0])
            logger.info(f"First embedding L2 norm: {first_norm}")
            
            if abs(first_norm - 1.0) > 0.01:  # If not already normalized
                logger.info("Normalizing embeddings")
                stored_embeddings_array = normalize(stored_embeddings_array)
                query_embedding_array = normalize(query_embedding_array)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding_array, stored_embeddings_array)[0]
            
            # Log some similarity values for debugging
            logger.info(f"Similarity range: {np.min(similarities):.4f} to {np.max(similarities):.4f}")
            logger.info(f"Top 5 similarities: {sorted(similarities, reverse=True)[:5]}")
            
            # Get indices of top N similar items
            top_indices = np.argsort(similarities)[::-1][:top_n]
            
            # Create result list
            results = []
            for idx in top_indices:
                if idx < len(file_paths):  # Ensure index is valid
                    results.append({
                        'file_path': file_paths[idx],
                        'image_url': get_image_url(file_paths[idx]),
                        'similarity': float(similarities[idx])
                    })
                
            logger.info(f"Found {len(results)} similar images")
            return results
        except Exception as e:
            logger.error(f"Error during similarity calculation: {str(e)}", exc_info=True)
            return []
        
    except Exception as e:
        logger.error(f"Error finding similar images: {str(e)}", exc_info=True)
        return [] 