import logging
from app.services.file_service import save_uploaded_file

logger = logging.getLogger(__name__)

def process_ai_request(request, model_name):
    """
    Common function to process AI requests with different models
    
    Args:
        request: The Flask request object
        model_name: Name of the AI model to use
        
    Returns:
        dict: Response data
    """
    # Get text from form data
    text = request.form.get('text', '')
    logger.info(f"Received text for {model_name}: {text}")
    
    # Process image if present
    image_path = None
    if 'image' in request.files:
        image = request.files['image']
        if image.filename:
            logger.info(f"Received image for {model_name}: {image.filename}")
            image_path = save_uploaded_file(image)
    
    # Here you would typically process the request with the specific AI model
    # This would be implemented differently for each model
    
    # For now, return a placeholder response
    return {
        'message': f'Successfully processed your {model_name} request. Text: "{text}"',
        'image_received': image_path is not None,
        'image_path': image_path,
        'model': model_name
    } 