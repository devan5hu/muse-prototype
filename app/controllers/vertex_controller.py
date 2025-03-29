import logging
from flask import request, jsonify, make_response
from app.services.file_service import save_uploaded_file
from app.utils.helpers import handle_options_request
from app.services.vertex_service import get_vertex_embeddings

logger = logging.getLogger(__name__)

def handle_vertex_request():
    """Handle Vertex AI requests with image and text for embeddings"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    logger.info("Request received at /vertex/chat")
    logger.info(f"Request method: {request.method}")
    
    try:
        # Get text from form data
        text = request.form.get('text', '')
        logger.info(f"Received text for Vertex: {text}")
        
        # Process image
        image_path = None
        if 'image' in request.files:
            image = request.files['image']
            if image.filename:
                logger.info(f"Received image for Vertex: {image.filename}")
                image_path = save_uploaded_file(image)
                logger.info(f"Image saved to: {image_path}")
            else:
                return create_cors_response({'error': 'Empty image file received'}, 400)
        else:
            return create_cors_response({'error': 'No image provided'}, 400)
        
        if not text:
            return create_cors_response({'error': 'No text provided'}, 400)
            
        # Get embeddings from Vertex AI
        embeddings = get_vertex_embeddings(image_path, text)
        
        # Create response
        response_data = {
            'message': 'Successfully processed with Vertex AI',
            'text': text,
            'image_path': image_path,
            'embeddings': embeddings
        }
        
        return create_cors_response(response_data)
    
    except Exception as e:
        logger.error(f"Error processing Vertex request: {str(e)}", exc_info=True)
        return create_cors_response({'error': str(e)}, 500)

def create_cors_response(data, status_code=200):
    """Create a response with CORS headers"""
    response = jsonify(data)
    response.status_code = status_code
    return response 