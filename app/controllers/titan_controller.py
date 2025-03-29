import logging
import os
from flask import request, jsonify
from app.utils.helpers import handle_options_request
from app.services.ai_service import process_ai_request
from app.services.titan_service import (
    get_titan_image_embedding,
    get_titan_text_embedding,
    get_titan_multimodal_embedding
)
from app.services.file_service import save_uploaded_file
from app.services.similarity_service import find_similar_images

logger = logging.getLogger(__name__)

def handle_titan_request():
    """Handle Titan AI requests with optional image uploads"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    logger.info("Request received at /titan")
    
    try:
        response_data = process_ai_request(request, 'titan')
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error processing Titan request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500 

def process_titan_embedding():
    """
    Process a request for Titan embeddings and find similar images
    
    Request can include:
    - image file
    - text
    - or both for multimodal embedding
    
    Returns:
        JSON response with embeddings and similar images
    """
    try:
        # Check if image file is provided
        image_file = request.files.get('image')
        text = request.form.get('text', '')
        
        # Determine the embedding type based on inputs
        if image_file and text:
            # Multimodal embedding (image + text)
            logger.info("Processing multimodal embedding request")
            
            # Save the uploaded image
            image_path = save_uploaded_file(image_file)
            
            # Get multimodal embeddings
            result = get_titan_multimodal_embedding(image_path, text)
            embedding_type = "multimodal"
            
        elif image_file:
            # Image-only embedding
            logger.info("Processing image-only embedding request")
            
            # Save the uploaded image
            image_path = save_uploaded_file(image_file)
            
            # Get image embeddings
            result = get_titan_image_embedding(image_path)
            embedding_type = "image"
            
        elif text:
            # Text-only embedding
            logger.info("Processing text-only embedding request")
            
            # Get text embeddings
            result = get_titan_text_embedding(text)
            embedding_type = "text"
            
        else:
            # No valid input provided
            logger.error("No image or text provided for embedding")
            return jsonify({
                "success": False,
                "message": "Please provide an image, text, or both for embedding generation",
                "embedding": None,
                "embedding_type": None,
                "similar_images": []
            }), 400
        
        # Check if we got a valid embedding
        if "error" not in result and result.get("embedding"):
            # Find similar images
            similar_images = find_similar_images(result["embedding"])
            
            # Log the results to console
            logger.info("Top 10 similar images:")
            for i, img in enumerate(similar_images):
                logger.info(f"{i+1}. {img['file_path']} (similarity: {img['similarity']:.4f})")
            
            # Format the response for the frontend to easily display images
            formatted_images = []
            for img in similar_images:
                formatted_images.append({
                    'url': img['image_url'],
                    'similarity': round(img['similarity'] * 100, 2),  # Convert to percentage
                    'filename': os.path.basename(img['file_path'])
                })
            
            return jsonify({
                "success": True,
                "message": f"Titan {embedding_type} embedding generated successfully",
                "embedding": result.get("embedding"),
                "embedding_type": embedding_type,
                "similar_images": similar_images,
                "formatted_images": formatted_images,
                "static_url": "/static/all_images/"
            })
        else:
            return jsonify({
                "success": False,
                "message": result.get("error", "Unknown error generating embedding"),
                "embedding": result.get("embedding"),
                "embedding_type": embedding_type,
                "similar_images": []
            })
            
    except Exception as e:
        logger.error(f"Error processing Titan embedding request: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error processing request: {str(e)}",
            "embedding": None,
            "embedding_type": None,
            "similar_images": []
        }), 500 