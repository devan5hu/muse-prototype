import os
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from app.services.titan_service import get_titan_embedding
from app.services.similarity_service import load_embeddings

logger = logging.getLogger(__name__)

def search_multimodal(query_text, query_image_path=None, top_k=10, image_weight=0.5):
    """
    Search using text query and optionally an image query
    
    Args:
        query_text: Text query (required)
        query_image_path: Path to query image (optional)
        top_k: Number of results to return
        image_weight: Weight for image similarity (ignored if no image provided)
    """
    try:
        # Get text embedding
        try:
            text_result = get_titan_embedding(text=query_text)
            text_embedding = text_result["embedding"]
        except Exception as e:
            logger.error(f"Failed to generate text embedding: {str(e)}")
            return []
        
        # Load embeddings data
        embeddings_dict = load_embeddings()
        if not embeddings_dict:
            logger.error("Could not load embeddings data")
            return []
        
        # Extract paths and embeddings from dictionary
        paths = []
        image_embeddings = []
        
        for image_id, data in embeddings_dict.items():
            if isinstance(data, dict) and 'embedding' in data:
                paths.append(data.get('path', image_id))
                image_embeddings.append(data['embedding'])
        
        embeddings_array = np.array(image_embeddings)
        query_text_array = np.array(text_embedding).reshape(1, -1)
        
        # Calculate text similarities
        text_similarities = cosine_similarity(query_text_array, embeddings_array)[0]
        
        # If image path is provided, include image similarity
        image_embedding = None
        image_similarities = None
        
        if query_image_path:
            try:
                image_result = get_titan_embedding(image_path=query_image_path)
                image_embedding = image_result["embedding"]
                query_image_array = np.array(image_embedding).reshape(1, -1)
                image_similarities = cosine_similarity(query_image_array, embeddings_array)[0]
                
                # Combine similarities with weights
                combined_similarities = (image_weight * image_similarities + 
                                      (1 - image_weight) * text_similarities)
                
                logger.info(f"Using combined text and image search with weight {image_weight}")
            except Exception as e:
                logger.warning(f"Failed to generate image embedding, using text-only search: {str(e)}")
                combined_similarities = text_similarities
        else:
            # Text-only search
            combined_similarities = text_similarities
            logger.info("Using text-only search")
        
        # Get top-k indices
        top_indices = np.argsort(combined_similarities)[::-1][:top_k]
        
        # Format results
        results = []
        for i in top_indices:
            if i < len(paths):  # Ensure index is valid
                result = {
                    'file_path': paths[i],
                    'image_url': f"/static/all_images/{os.path.basename(paths[i])}",
                    'combined_similarity': float(combined_similarities[i]),
                    'text_similarity': float(text_similarities[i])
                }
                
                # Add image similarity if available
                if image_similarities is not None:
                    result['image_similarity'] = float(image_similarities[i])
                
                results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in multimodal search: {str(e)}", exc_info=True)
        return [] 