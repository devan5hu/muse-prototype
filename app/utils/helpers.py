from flask import jsonify, make_response
import logging

logger = logging.getLogger(__name__)

def allowed_file(filename, allowed_extensions=None):
    """Check if file has an allowed extension"""
    if allowed_extensions is None:
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

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
    response.headers.add('Access-Control-Allow-Origin', '*')  # For development, allow all origins
    # For production, you might want to be more specific:
    # response.headers.add('Access-Control-Allow-Origin', 'https://67e7c87e9879df0035944485--peaceful-bonbon-e63768.netlify.app')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response 