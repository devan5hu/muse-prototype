import os
import json
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.services.voyage_service import get_voyage_embedding
from app.utils.helpers import allowed_file, create_cors_response, handle_options_request
from PIL import Image

voyage_bp = Blueprint('voyage', __name__)

@voyage_bp.route('/api/voyage/search', methods=['POST', 'OPTIONS'])
def search():
    """Search for similar images using Voyage embeddings from a static pickle file."""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    query_text = None
    query_image_path = None
    image_weight = 0.4  # Default weight for image similarity
    img = None 
    # Debug the incoming request
    print(f"Request content type: {request.content_type}")
    
    # Check if the request contains form data or JSON
    if request.content_type and 'multipart/form-data' in request.content_type:
        # Handle form data
        query_text = request.form.get('text')  # Get text, could be None
        query_image = request.files.get('image')
        
        # Get image weight if provided
        try:
            weight = request.form.get('image_weight')
            if weight is not None:
                image_weight = float(weight)
        except:
            pass
            
        print(f"Form data - text: {query_text}, image: {query_image}, image_weight: {image_weight}")
        
        # If an image was uploaded, save it temporarily
        if query_image and query_image.filename:
            # Print image details for debugging
            print(f"Received image: {query_image.filename}, content_type: {query_image.content_type}")
            
            filename = secure_filename(query_image.filename)
            upload_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], 'temp')
            os.makedirs(upload_folder, exist_ok=True)
            query_image_path = os.path.join(upload_folder, filename)
            query_image.save(query_image_path)
            print(f"Saved image to: {query_image_path}")
            
            # Verify the saved file
            try:
                img = Image.open(query_image_path)
                print(f"Image details: format={img.format}, mode={img.mode}, size={img.size}")
            except Exception as e:
                print(f"Error verifying saved image: {str(e)}")
    else:
        # Handle JSON data
        data = request.json
        if data:
            query_text = data.get('query')
            query_image_path = data.get('query_image_path')
            
            # Get image weight if provided
            if 'image_weight' in data:
                try:
                    image_weight = float(data['image_weight'])
                except:
                    pass
                    
            print(f"JSON data - query: {query_text}, image_path: {query_image_path}, image_weight: {image_weight}")
    
    # Ensure at least one of text or image is provided
    if not query_text and not query_image_path:
        return jsonify({'error': 'No query text or image provided'}), 400
    
    # Load pre-computed embeddings from static pickle file
    import pickle
    embeddings_file = os.path.join(current_app.root_path, 'static', 'json', 'emb_selected_images.pkl')
    
    # Try alternative paths if the file is not found
    alt_path1 = os.path.join('static', 'json', 'emb_selected_images.pkl')
    alt_path2 = os.path.join(os.getcwd(), 'static', 'json', 'emb_selected_images.pkl')
    alt_path3 = os.path.join(os.getcwd(), 'app', 'static', 'json', 'emb_selected_images.pkl')
    
    # Try to find the file in multiple locations
    if os.path.exists(embeddings_file):
        embeddings_path = embeddings_file
    elif os.path.exists(alt_path1):
        embeddings_path = alt_path1
    elif os.path.exists(alt_path2):
        embeddings_path = alt_path2
    elif os.path.exists(alt_path3):
        embeddings_path = alt_path3
    else:
        return jsonify({'error': f'Embeddings file not found. Tried: {embeddings_file}, {alt_path1}, {alt_path2}, {alt_path3}'}), 404
    
    try:
        # Load pickle file
        with open(embeddings_path, 'rb') as f:
            stored_data = pickle.load(f)
        
        # Print structure for debugging
        print(f"Pickle structure: {type(stored_data)}")
        if isinstance(stored_data, dict):
            print(f"Keys: {stored_data.keys()}")
        
        # Generate embedding for the query
        query_embedding = None
        if query_text and query_image_path:
            # Both text and image
            query_embedding = get_voyage_embedding(text=query_text, img=img)
        elif query_image_path:
            # Only image
            query_embedding = get_voyage_embedding(img=img)
        elif query_text:
            # Only text
            query_embedding = get_voyage_embedding(text=query_text)
        
        # Compute similarities
        results = []
        
        # Handle different data structures
        embeddings = []
        image_paths = []
        
        if isinstance(stored_data, dict):
            if 'embeddings' in stored_data and 'image_paths' in stored_data:
                # Format: {"embeddings": [...], "image_paths": [...]}
                embeddings = stored_data['embeddings']
                image_paths = stored_data['image_paths']
            elif 'emb' in stored_data and 'paths' in stored_data:
                # Alternative format
                embeddings = stored_data['emb']
                image_paths = stored_data['paths']
        
        # Compute similarity for each embedding
        for idx, embedding_item in enumerate(embeddings):
            if idx >= len(image_paths):
                break
                
            img_path = image_paths[idx]
            
            # Ensure embeddings are numpy arrays
            embedding_array = np.array(embedding_item, dtype=np.float32)
            query_array = np.array(query_embedding, dtype=np.float32)
            
            # Compute similarity
            try:
                similarity = float(np.dot(query_array, embedding_array) / 
                                 (np.linalg.norm(query_array) * np.linalg.norm(embedding_array)))
            except Exception as e:
                print(f"Error computing similarity: {e}")
                similarity = 0.0
            
            # Extract just the filename from the path
            filename = os.path.basename(img_path)
            
            results.append({
                'image_path': filename,  # Just the filename
                'full_path': img_path,   # Keep the full path for debugging
                'similarity': similarity
            })
        
        # Sort by similarity (descending) and get top 10
        results.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = results[:10]
        
        # Format the response to match what the frontend expects
        formatted_images = []
        for result in top_results:
            formatted_images.append({
                'url': result['image_path'],  # Just the filename
                'similarity': result['similarity']
            })
        
        result = {
            'success': True,
            'formatted_images': formatted_images,
            'similar_images': formatted_images,
            'image_weight': image_weight
        }
        
        return create_cors_response(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary image file if it was created
        if query_image_path and os.path.exists(query_image_path) and 'temp' in query_image_path:
            try:
                os.remove(query_image_path)
            except:
                pass