import logging
from flask import request, jsonify
from app.services.file_service import save_uploaded_file
from app.utils.helpers import handle_options_request

logger = logging.getLogger(__name__)

def handle_azure_request():
    """Handle Azure AI requests with optional image uploads"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    logger.info("Request received at /azure")
    logger.info(f"Request method: {request.method}")
    
    try:
        # Get text from form data
        text = request.form.get('text', '')
        logger.info(f"Received text for Azure: {text}")
        
        # Process image if present
        image_path = None
        if 'image' in request.files:
            image = request.files['image']
            if image.filename:
                logger.info(f"Received image for Azure: {image.filename}")
                image_path = save_uploaded_file(image)
        
        # Here you would typically process the request with Azure AI
        # For now, we'll just return a placeholder response
        
        response_data = {
            'message': f'Successfully processed your Azure request. Text: "{text}"',
            'image_received': image_path is not None,
            'image_path': image_path,
            'model': 'azure'
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error processing Azure request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500 