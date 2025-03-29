import logging
import os
from flask import request, jsonify
from app.services.twelvelabs_service import search_multimodal, get_embedding_for_text, get_embedding_for_image
from app.services.file_service import save_uploaded_file
from app.utils.helpers import handle_options_request

logger = logging.getLogger(__name__)

def handle_twelvelabs_request():
    """Handle TwelveLabs requests with optional image uploads"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    logger.info("Request received at /twelvelabs")
    logger.info(f"Request method: {request.method}")
    
    try:
        # Get text from form data
        text = request.form.get('text', '')
        logger.info(f"Received text for TwelveLabs: {text}")
        
        # Process image if present
        image_path = None
        if 'image' in request.files:
            image = request.files['image']
            if image.filename:
                logger.info(f"Received image for TwelveLabs: {image.filename}")
                image_path = save_uploaded_file(image)
        
        # Here you would typically process the request with TwelveLabs
        # For now, we'll just return a placeholder response
        
        response_data = {
            'message': f'Successfully processed your TwelveLabs request. Text: "{text}"',
            'image_received': image_path is not None,
            'image_path': image_path,
            'model': 'twelvelabs'
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error processing TwelveLabs request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def handle_twelvelabs_search():
    """
    Handle multimodal search requests with text and/or image
    
    This function can handle:
    1. Text-only search (when only 'text' is provided)
    2. Image-only search (when only 'image' is provided)
    3. Combined text+image search (when both are provided)
    """
    try:
        # Get query text
        query_text = request.form.get('text', '')
        
        # Get optional image
        image_file = request.files.get('image')
        image_path = None
        if image_file and image_file.filename:
            try:
                image_path = save_uploaded_file(image_file)
                logger.info(f"Saved query image to {image_path}")
            except Exception as e:
                logger.error(f"Error saving image: {str(e)}")
                return jsonify({
                    "success": False,
                    "message": f"Error processing image: {str(e)}",
                    "results": []
                }), 400
        
        # Validate inputs - at least one of text or image must be provided
        if not query_text and not image_path:
            return jsonify({
                "success": False,
                "message": "Either text or image query (or both) is required",
                "results": []
            }), 400
        
        # Get optional parameters
        top_k = int(request.form.get('top_k', 10))
        
        # Set image weight based on what's provided
        if query_text and image_path:
            # Both text and image provided - use the weight from request or default
            image_weight = float(request.form.get('image_weight', 0.4))
            logger.info(f"Performing combined text+image search with weight {image_weight}")
        elif image_path:
            # Only image provided - set weight to 1.0
            image_weight = 1.0
            logger.info("Performing image-only search")
        else:
            # Only text provided - set weight to 0.0
            image_weight = 0.0
            logger.info("Performing text-only search")
        
        # Perform search
        results = search_multimodal(
            query_text=query_text,
            query_image_path=image_path,
            top_k=top_k,
            image_weight=image_weight
        )
        
        # Format results for frontend display
        formatted_results = []
        for result in results:
            formatted_result = {
                'url': result['image_url'],
                'filename': os.path.basename(result['file_path']),
                'combined_similarity': round(result['combined_similarity'] * 100, 2)
            }
            
            if 'text_similarity' in result:
                formatted_result['text_similarity'] = round(result['text_similarity'] * 100, 2)
                
            if 'image_similarity' in result:
                formatted_result['image_similarity'] = round(result['image_similarity'] * 100, 2)
                
            formatted_results.append(formatted_result)
        
        # Determine search type for response
        search_type = "text"
        if query_text and image_path:
            search_type = "multimodal"
        elif image_path:
            search_type = "image"
        
        return jsonify({
            "success": True,
            "message": "Search completed successfully",
            "search_type": search_type,
            "results": results,
            "formatted_results": formatted_results
        })
        
    except Exception as e:
        logger.error(f"Error in Twelve Labs search: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}",
            "results": []
        }), 500

def handle_twelvelabs_embedding():
    """Generate embeddings using Twelve Labs API"""
    try:
        # Get embedding type
        embedding_type = request.form.get('type', 'text')
        
        if embedding_type == 'text':
            # Get text
            text = request.form.get('text')
            if not text:
                return jsonify({
                    "success": False,
                    "message": "Text is required for text embeddings",
                    "embedding": None
                }), 400
                
            # Generate embedding
            embedding = get_embedding_for_text(text)
            
            return jsonify({
                "success": True,
                "message": "Text embedding generated successfully",
                "embedding": embedding,
                "embedding_type": "text"
            })
            
        elif embedding_type == 'image':
            # Get image
            image_file = request.files.get('image')
            if not image_file:
                return jsonify({
                    "success": False,
                    "message": "Image file is required for image embeddings",
                    "embedding": None
                }), 400
                
            # Save image
            image_path = save_uploaded_file(image_file)
            
            # Generate embedding
            embedding = get_embedding_for_image(image_path)
            
            return jsonify({
                "success": True,
                "message": "Image embedding generated successfully",
                "embedding": embedding,
                "embedding_type": "image",
                "image_path": image_path
            })
            
        else:
            return jsonify({
                "success": False,
                "message": f"Unsupported embedding type: {embedding_type}",
                "embedding": None
            }), 400
            
    except Exception as e:
        logger.error(f"Error generating Twelve Labs embedding: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}",
            "embedding": None
        }), 500 