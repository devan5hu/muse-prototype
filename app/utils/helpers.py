from flask import jsonify, make_response, current_app
import logging
import os
import uuid

logger = logging.getLogger(__name__)

def allowed_file(filename, allowed_extensions=None):
    """Check if file has an allowed extension"""
    if allowed_extensions is None:
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_uploaded_file(file):
    """
    Save an uploaded file to the upload folder with a unique filename
    
    Args:
        file: The file object from request.files
        
    Returns:
        str: Path to the saved file
    """
    # Generate a unique filename to avoid collisions
    original_filename = file.filename
    extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
    unique_filename = f"{uuid.uuid4().hex}.{extension}" if extension else f"{uuid.uuid4().hex}"
    
    # Create upload directory if it doesn't exist
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
    os.makedirs(upload_folder, exist_ok=True)
    
    # Save the file
    file_path = os.path.join(upload_folder, unique_filename)
    file.save(file_path)
    logger.info(f"Saved uploaded file to {file_path}")
    
    return file_path

def handle_options_request():
    """Handle CORS preflight requests"""
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

def handle_404_error(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

def handle_413_error(error):
    """Handle file too large errors"""
    return jsonify({'error': 'File too large'}), 413

def create_cors_response(response=None, status=200):
    """Create a response with CORS headers"""
    if response is None:
        response = jsonify({})
        
    if isinstance(response, tuple) and len(response) == 2:
        response, status = response
        
    response = make_response(response, status)
    # Use '*' for development or specific origins for production
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response 

def get_file_url(file_path):
    """
    Convert a local file path to a URL that can be accessed externally.
    This is needed for Azure API which requires URLs instead of file objects.
    
    Args:
        file_path (str): Local file path
        
    Returns:
        str: URL that can be accessed by external services
    """
    # In a production environment, this would return a publicly accessible URL
    # For local development, you might need to use a service like ngrok
    # or upload the file to a cloud storage service
    
    # For now, we'll use a placeholder that should be replaced in production
    base_url = os.environ.get("BASE_URL", "http://localhost:5000")
    
    # Convert the file path to a URL path
    if file_path.startswith("static/"):
        url_path = file_path
    else:
        url_path = os.path.join("static", file_path)
    
    # Replace backslashes with forward slashes for URLs
    url_path = url_path.replace("\\", "/")
    
    # Combine base URL with file path
    return f"{base_url}/{url_path}" 