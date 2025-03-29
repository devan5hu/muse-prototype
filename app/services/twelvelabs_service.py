import os
import logging
import time
import numpy as np
from twelvelabs import TwelveLabs
from sklearn.metrics.pairwise import cosine_similarity
import json
from flask import current_app

logger = logging.getLogger(__name__)

# Configuration
TWELVELABS_API_KEY = os.environ.get("TWELVELABS_API_KEY")
TWELVELABS_EMBEDDINGS_JSON = "static/json/twelve_labs_embeddings.json"

# Initialize client
client = None
try:
    client = TwelveLabs(api_key=TWELVELABS_API_KEY)
    logger.info("Twelve Labs client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Twelve Labs client: {str(e)}")

def get_embedding_for_text(text):
    """Generate embedding for text using Twelve Labs API."""
    try:
        if client is None:
            logger.error("Twelve Labs client not initialized")
            raise RuntimeError("Twelve Labs client not initialized")
            
        logger.info(f"Generating text embedding for: {text[:50]}...")
        response = client.embed.create(text=text, model_name="Marengo-retrieval-2.7")
        
        if response.text_embedding and response.text_embedding.segments:
            embedding = response.text_embedding.segments[0].embeddings_float
            logger.info(f"Successfully generated text embedding of length {len(embedding)}")
            return embedding
        else:
            logger.error(f"Could not find embedding in response for text: {text}")
            raise ValueError("No embedding found in response")
            
    except Exception as e:
        logger.error(f"Error generating text embedding: {str(e)}", exc_info=True)
        raise

def get_embedding_for_image(image_path):
    """Generate embedding for an image using Twelve Labs API."""
    try:
        if client is None:
            logger.error("Twelve Labs client not initialized")
            raise RuntimeError("Twelve Labs client not initialized")
            
        logger.info(f"Generating image embedding for: {image_path}")
        
        with open(image_path, 'rb') as img_file:
            response = client.embed.create(image_file=img_file, model_name="Marengo-retrieval-2.7")
        
        if response.image_embedding and response.image_embedding.segments:
            embedding = response.image_embedding.segments[0].embeddings_float
            logger.info(f"Successfully generated image embedding of length {len(embedding)}")
            return embedding
        else:
            logger.error(f"Could not find embedding in response for {image_path}")
            raise ValueError("No embedding found in response")
            
    except Exception as e:
        logger.error(f"Error generating image embedding: {str(e)}", exc_info=True)
        raise

def load_embeddings():
    """Load embeddings from JSON file"""
    try:
        if not os.path.exists(TWELVELABS_EMBEDDINGS_JSON):
            logger.error(f"Embeddings file not found: {TWELVELABS_EMBEDDINGS_JSON}")
            return None
            
        with open(TWELVELABS_EMBEDDINGS_JSON, 'r') as f:
            embeddings_data = json.load(f)
            
        logger.info(f"Loaded embeddings from JSON file: {len(embeddings_data.get('image_paths', []))} entries")
        return embeddings_data
        
    except Exception as e:
        logger.error(f"Error loading embeddings from JSON: {str(e)}", exc_info=True)
        return None

def search_multimodal(query_text=None, query_image_path=None, top_k=7, image_weight=0.5):
    """
    Search for similar images using text and/or image queries.
    
    This function can handle:
    1. Text-only search (when only query_text is provided)
    2. Image-only search (when only query_image_path is provided)
    3. Combined text+image search (when both are provided)
    
    Args:
        query_text: Text description to search for (optional if image is provided)
        query_image_path: Path to query image (optional if text is provided)
        top_k: Number of top results to return
        image_weight: Weight for image similarity (0-1), text weight will be (1-image_weight)
        
    Returns:
        List of dictionaries with image paths and similarity scores
    """
    try:
        # Validate inputs
        if not query_text and not query_image_path:
            logger.error("Both query_text and query_image_path cannot be None")
            raise ValueError("Either query_text or query_image_path must be provided")
            
        with open(TWELVELABS_EMBEDDINGS_JSON, 'r') as f:
            embeddings_dict = json.load(f)
            
        if not embeddings_dict:
            logger.error("Embeddings dictionary is empty")
            raise ValueError("No embeddings found in the embeddings file")
            
        # Extract image paths and embeddings
        image_paths = list(embeddings_dict.keys())
        embeddings = [embeddings_dict[img_path] for img_path in image_paths]
        
        # Check if we have valid data
        if not image_paths or not embeddings:
            logger.error("No valid image paths or embeddings found")
            raise ValueError("No valid image paths or embeddings found")
            
        logger.info(f"Loaded {len(image_paths)} embeddings from file")
        
        # Convert embeddings to numpy array
        try:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            logger.info(f"Embeddings array shape: {embeddings_array.shape}")
        except Exception as e:
            logger.error(f"Error converting embeddings to numpy array: {str(e)}")
            
            # Try to handle inhomogeneous arrays by filtering
            valid_embeddings = []
            valid_image_paths = []
            embedding_length = None
            
            # Check embedding dimensions and filter out invalid ones
            for idx, emb in enumerate(embeddings):
                try:
                    # Convert to numpy array to check shape
                    emb_array = np.array(emb, dtype=np.float32)
                    
                    # Set expected length if not set
                    if embedding_length is None:
                        embedding_length = len(emb_array)
                        
                    # Only keep embeddings with the expected length
                    if len(emb_array) == embedding_length:
                        valid_embeddings.append(emb_array)
                        valid_image_paths.append(image_paths[idx])
                except Exception as inner_e:
                    logger.warning(f"Skipping invalid embedding for {image_paths[idx]}: {str(inner_e)}")
                    
            if not valid_embeddings:
                logger.error("No valid embeddings found after filtering")
                raise ValueError("No valid embeddings found after filtering")
                
            embeddings_array = np.array(valid_embeddings, dtype=np.float32)
            image_paths = valid_image_paths
            logger.info(f"After filtering: {len(valid_image_paths)} valid embeddings with shape {embeddings_array.shape}")
        
        # Initialize similarity scores
        text_similarities = None
        image_similarities = None
        
        # Text-based search
        if query_text:
            logger.info(f"Performing text search with query: {query_text}")
            text_embedding = get_embedding_for_text(query_text)
            text_embedding_array = np.array(text_embedding, dtype=np.float32).reshape(1, -1)
            text_similarities = cosine_similarity(text_embedding_array, embeddings_array)[0]
            logger.info(f"Text similarity range: {np.min(text_similarities):.4f} to {np.max(text_similarities):.4f}")
        
        # Image-based search
        if query_image_path:
            logger.info(f"Performing image search with image: {query_image_path}")
            image_embedding = get_embedding_for_image(query_image_path)
            image_embedding_array = np.array(image_embedding, dtype=np.float32).reshape(1, -1)
            image_similarities = cosine_similarity(image_embedding_array, embeddings_array)[0]
            logger.info(f"Image similarity range: {np.min(image_similarities):.4f} to {np.max(image_similarities):.4f}")
        
        # Combine similarities if both text and image queries are provided
        if query_text and query_image_path:
            logger.info(f"Combining text and image similarities with image_weight={image_weight}")
            combined_similarities = (1 - image_weight) * text_similarities + image_weight * image_similarities
        elif query_text:
            combined_similarities = text_similarities
        else:
            combined_similarities = image_similarities
        
        # Get top-k indices
        top_indices = np.argsort(combined_similarities)[::-1][:top_k]
        
        # Create results
        results = []
        for idx in top_indices:
            if idx >= len(image_paths):
                continue
                
            img_path = image_paths[idx]
            
            result = {
                'file_path': img_path,
                'image_url': f"/static/all_images/{img_path}",
                'combined_similarity': float(combined_similarities[idx])
            }
            
            # Add individual similarities if available
            if text_similarities is not None:
                result['text_similarity'] = float(text_similarities[idx])
                
            if image_similarities is not None:
                result['image_similarity'] = float(image_similarities[idx])
                
            results.append(result)
        
        # Sort by combined similarity
        results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        
        # Return top-k results
        top_results = results[:top_k]
        logger.info(f"Found {len(top_results)} results for search")
        return top_results
        
    except Exception as e:
        logger.error(f"Error in multimodal search: {str(e)}", exc_info=True)
        raise 