from flask import Blueprint, request, jsonify
from app.services.titan_service import get_titan_embedding, get_titan_text_embedding
from app.utils.helpers import create_cors_response
from app.services.file_service import save_uploaded_file
from app.services.similarity_service import find_similar_images
import logging
import os
import json
import base64
import tempfile
from urllib.parse import urlparse
import traceback

logger = logging.getLogger(__name__)

# Create a Blueprint for Titan routes with a prefix
titan_bp = Blueprint('titan', __name__, url_prefix='/titan')

@titan_bp.route('/embedding', methods=['POST', 'OPTIONS'])
def embedding():
    """Route for Titan embedding endpoint"""
    if request.method == 'OPTIONS':
        return create_cors_response()
    
    try:
        # Log EVERYTHING for debugging
        logger.info("==== DETAILED REQUEST INFO ====")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request has files: {bool(request.files)}")
        logger.info(f"Request is JSON: {request.is_json}")
        logger.info(f"Form data keys: {list(request.form.keys())}")
        logger.info(f"Form data values: {dict(request.form)}")
        logger.info(f"Files: {list(request.files.keys())}")
        
        # Check for image in request
        if 'image' in request.files:
            file = request.files['image']
            logger.info(f"Found image file: {file.filename}")
            file_path = save_uploaded_file(file)
            logger.info(f"Saved image to: {file_path}")
            
            # Get image embedding
            result = get_titan_embedding(image_path=file_path)
            embedding_type = "image"
            
        # Check for file in request (alternative name)
        elif 'file' in request.files:
            file = request.files['file']
            logger.info(f"Found file: {file.filename}")
            file_path = save_uploaded_file(file)
            logger.info(f"Saved file to: {file_path}")
            
            # Get image embedding
            result = get_titan_embedding(image_path=file_path)
            embedding_type = "image"
            
        # Check for text in request
        elif 'text' in request.form and request.form.get('text').strip():
            text = request.form.get('text').strip()
            logger.info(f"Found text: {text}")
            
            # Get text embedding
            result = get_titan_text_embedding(text)
            embedding_type = "text"
            
        # If we have neither valid image nor text, return error
        else:
            logger.error("No valid image or text found in request")
            return create_cors_response(jsonify({
                'error': 'No valid image or text found in request',
                'request_details': {
                    'content_type': request.content_type,
                    'has_files': bool(request.files),
                    'files': list(request.files.keys()) if request.files else [],
                    'form_data': dict(request.form),
                }
            }), 400)
        
        # Find similar images
        similar_images = find_similar_images(result["embedding"])
        
        # Format response
        formatted_images = []
        for img in similar_images:
            formatted_images.append({
                'url': img['image_url'],
                'similarity': round(img['similarity'] * 100, 2),
                'filename': os.path.basename(img['file_path'])
            })
        
        # Create response
        response_data = {
            'success': True,
            'embedding': result['embedding'],
            'embedding_type': embedding_type,
            'similar_images': similar_images,
            'formatted_images': formatted_images,
            'static_url': "/static/all_images/"
        }
        
        # Add specific data
        if embedding_type == "image":
            response_data['file_path'] = file_path
            response_data['image_url'] = f"/static/uploads/{os.path.basename(file_path)}"
        elif embedding_type == "text":
            response_data['text'] = request.form.get('text')
        
        return create_cors_response(jsonify(response_data))
        
    except Exception as e:
        logger.error(f"Error in Titan embedding endpoint: {str(e)}", exc_info=True)
        return create_cors_response(jsonify({'error': str(e), 'traceback': str(traceback.format_exc())}), 500) 