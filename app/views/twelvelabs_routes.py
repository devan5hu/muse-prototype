from flask import Blueprint, request, jsonify
from app.controllers.twelvelabs_controller import handle_twelvelabs_search, handle_twelvelabs_embedding
from app.utils.helpers import handle_options_request, create_cors_response
from app.services.twelvelabs_service import search_multimodal
from app.services.file_service import save_uploaded_file
import logging
import os
import traceback
import base64
import tempfile

logger = logging.getLogger(__name__)

twelvelabs_bp = Blueprint('twelvelabs', __name__, url_prefix='/twelvelabs')

@twelvelabs_bp.route('/search', methods=['POST', 'OPTIONS'])
def search():
    """Route for multimodal search using Twelve Labs"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return create_cors_response()
        
    try:
        # Log EVERYTHING for debugging
        logger.info("==== DETAILED TWELVE LABS REQUEST INFO ====")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request has files: {bool(request.files)}")
        logger.info(f"Request is JSON: {request.is_json}")
        logger.info(f"Form data keys: {list(request.form.keys())}")
        logger.info(f"Form data values: {dict(request.form)}")
        logger.info(f"Files: {list(request.files.keys())}")
        
        # Initialize variables
        query_text = None
        query_image_path = None
        
        # Check for image in request
        if 'image' in request.files:
            file = request.files['image']
            logger.info(f"Found image file: {file.filename}")
            query_image_path = save_uploaded_file(file)
            logger.info(f"Saved image to: {query_image_path}")
            
        # Check for file in request (alternative name)
        elif 'file' in request.files:
            file = request.files['file']
            logger.info(f"Found file: {file.filename}")
            query_image_path = save_uploaded_file(file)
            logger.info(f"Saved file to: {query_image_path}")
            
        # Get text from form data or JSON
        if request.form and 'text' in request.form:
            query_text = request.form.get('text').strip()
            logger.info(f"Found text in form: {query_text}")
        elif request.is_json:
            try:
                data = request.get_json()
                if data:
                    query_text = data.get('query_text') or data.get('text')
                    if not query_image_path:
                        query_image_path = data.get('query_image_path') or data.get('image_path')
                    logger.info(f"Found text in JSON: {query_text}")
                    logger.info(f"Found image path in JSON: {query_image_path}")
            except Exception as e:
                logger.error(f"Error parsing JSON: {str(e)}")
        
        # Get parameters
        top_k = 7
        image_weight = 0.5
        
        if request.is_json:
            data = request.get_json() or {}
            top_k = int(data.get('top_k', top_k))
            image_weight = float(data.get('image_weight', image_weight))
        
        logger.info(f"Search parameters: query_text={query_text}, query_image_path={query_image_path}, top_k={top_k}, image_weight={image_weight}")
        
        # Validate input
        if not query_text and not query_image_path:
            logger.error("No valid query text or image found")
            return create_cors_response(jsonify({
                'success': False,
                'message': 'No valid query text or image found',
                'error': 'Please provide either text or an image for search'
            }), 400)
            
        # Perform search
        results = search_multimodal(
            query_text=query_text,
            query_image_path=query_image_path,
            top_k=top_k,
            image_weight=image_weight
        )
        
        logger.info(f"Search returned {len(results)} results")
        
        # Format the results for the frontend
        formatted_results = []
        for img in results:
            formatted_results.append({
                'url': img['image_url'],
                'similarity': round(img['combined_similarity'] * 100, 2) if 'combined_similarity' in img else round(img['text_similarity'] * 100, 2),
                'filename': os.path.basename(img['file_path'])
            })
        
        # Create response matching what the frontend expects
        response_data = {
            'success': True,
            'message': f"Found {len(results)} similar images",
            'results': results,
            'formatted_results': formatted_results
        }
        
        return create_cors_response(jsonify(response_data))
        
    except Exception as e:
        logger.error(f"Error in Twelve Labs search endpoint: {str(e)}", exc_info=True)
        return create_cors_response(jsonify({
            'success': False,
            'message': str(e),
            'error': str(e),
            'traceback': str(traceback.format_exc())
        }), 500)

@twelvelabs_bp.route('/embedding', methods=['POST', 'OPTIONS'])
def embedding():
    """Route for Twelve Labs embedding generation"""
    if request.method == 'OPTIONS':
        return create_cors_response()
    return create_cors_response(handle_twelvelabs_embedding())

@twelvelabs_bp.route('/status', methods=['GET', 'OPTIONS'])
def status():
    """Check if Twelve Labs service is available"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return handle_options_request()
        
    from app.services.twelvelabs_service import client
    
    if client:
        return jsonify({
            "status": "available",
            "message": "Twelve Labs service is available"
        })
    else:
        return jsonify({
            "status": "unavailable",
            "message": "Twelve Labs service is not available"
        }), 503 