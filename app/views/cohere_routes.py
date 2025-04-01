import os
import json
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.services.cohere_service import get_cohere_embedding, search_images, get_text_embedding, cosine_similarity
from app.utils.helpers import allowed_file

cohere_bp = Blueprint('cohere', __name__)

@cohere_bp.route('/api/cohere/embed', methods=['POST'])
def embed_image():
    """Generate embedding for an uploaded image using Cohere API."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Save the file
        upload_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        
        try:
            embedding = get_cohere_embedding(filepath)
            
            # Save embedding to a JSON file
            embedding_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'embeddings')
            os.makedirs(embedding_dir, exist_ok=True)
            
            embedding_filename = os.path.splitext(filename)[0] + '_cohere.json'
            embedding_path = os.path.join(embedding_dir, embedding_filename)
            
            with open(embedding_path, 'w') as f:
                json.dump({
                    'embedding': embedding.tolist(),
                    'image_path': filepath
                }, f)
            
            return jsonify({
                'success': True,
                'message': 'Embedding generated successfully',
                'embedding_path': embedding_path,
                'image_path': filepath
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@cohere_bp.route('/api/cohere/search', methods=['POST'])
def search():
    """Search for similar images using Cohere embeddings from a static JSON file."""
    query = None
    query_image_path = None
    
    # Check if the request contains form data or JSON
    if request.content_type and 'multipart/form-data' in request.content_type:
        # Handle form data
        query = request.form.get('text')
        query_image = request.files.get('image')
        
        # If an image was uploaded, save it temporarily
        if query_image and query_image.filename:
            filename = secure_filename(query_image.filename)
            upload_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], 'temp')
            os.makedirs(upload_folder, exist_ok=True)
            query_image_path = os.path.join(upload_folder, filename)
            query_image.save(query_image_path)
    else:
        # Handle JSON data
        data = request.json
        if data:
            query = data.get('query')
            query_image_path = data.get('query_image_path')
    
    # Ensure at least one of text or image is provided
    if not query and not query_image_path:
        return jsonify({'error': 'No query text or image provided'}), 400
    
    # Load pre-computed embeddings from static JSON file
    embeddings_file = os.path.join(current_app.root_path, 'static', 'json', 'cohere_embeddings_selected_images.json')
    
    # Try alternative paths
    alt_path1 = os.path.join('static', 'json', 'cohere_embeddings_selected_images.json')
    alt_path2 = os.path.join(os.getcwd(), 'static', 'json', 'cohere_embeddings_selected_images.json')
    alt_path3 = os.path.join(os.getcwd(), 'app', 'static', 'json', 'cohere_embeddings_selected_images.json')
    
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
        with open(embeddings_path, 'r') as f:
            stored_data = json.load(f)
        
        # Print structure for debugging
        print(f"JSON structure: {type(stored_data)}")
        if isinstance(stored_data, dict):
            print(f"Keys: {stored_data.keys()}")
        
        # Generate text embedding for the query if text is provided
        query_text_embedding = None
        if query:
            query_text_embedding = get_text_embedding(query)
            print(f"Query text embedding shape: {np.array(query_text_embedding).shape}")
        
        # Generate image embedding for the query if image path is provided
        query_image_embedding = None
        if query_image_path and os.path.exists(query_image_path):
            query_image_embedding = get_cohere_embedding(query_image_path)
            print(f"Query image embedding shape: {np.array(query_image_embedding).shape}")
        
        # Compute similarities
        results = []
        image_weight = 0.6  # Weight for image similarity vs text similarity
        
        # Handle different JSON structures
        embeddings = []
        image_paths = []
        
        if isinstance(stored_data, dict) and 'embeddings' in stored_data and 'image_paths' in stored_data:
            # Format: {"embeddings": [...], "image_paths": [...]}
            embeddings = stored_data['embeddings']
            image_paths = stored_data['image_paths']
        elif isinstance(stored_data, list):
            # Format: [{"embedding": [...], "image_path": "..."}, ...]
            for item in stored_data:
                if isinstance(item, dict) and 'embedding' in item and 'image_path' in item:
                    embeddings.append(item['embedding'])
                    image_paths.append(item['image_path'])
        
        if not embeddings or len(embeddings) == 0:
            return jsonify({'error': 'No embeddings found in the file'}), 404
        
        print(f"First embedding shape: {np.array(embeddings[0]).shape}")
        
        for idx, embedding_item in enumerate(embeddings):
            if idx >= len(image_paths):
                break
                
            img_path = image_paths[idx]
            
            # Initialize similarities
            text_similarity = 0.0
            image_similarity = 0.0
            combined_similarity = 0.0
            
            # Ensure embeddings are numpy arrays with the same shape
            embedding_array = np.array(embedding_item, dtype=np.float32)
            
            # Compute text similarity if text query is provided
            if query_text_embedding is not None:
                query_text_array = np.array(query_text_embedding, dtype=np.float32)
                
                # Ensure dimensions match
                if embedding_array.shape != query_text_array.shape:
                    print(f"Warning: Shape mismatch - embedding: {embedding_array.shape}, query: {query_text_array.shape}")
                    # Try to reshape if possible
                    if embedding_array.size == query_text_array.size:
                        embedding_array = embedding_array.reshape(query_text_array.shape)
                
                try:
                    text_similarity = float(np.dot(query_text_array, embedding_array) / 
                                          (np.linalg.norm(query_text_array) * np.linalg.norm(embedding_array)))
                except Exception as e:
                    print(f"Error computing text similarity: {e}")
                    text_similarity = 0.0
                
                # If only text query, use text similarity as combined similarity
                if query_image_embedding is None:
                    combined_similarity = text_similarity
            
            # Compute image similarity if image query is provided
            if query_image_embedding is not None:
                query_image_array = np.array(query_image_embedding, dtype=np.float32)
                
                # Ensure dimensions match
                if embedding_array.shape != query_image_array.shape:
                    print(f"Warning: Shape mismatch - embedding: {embedding_array.shape}, query: {query_image_array.shape}")
                    # Try to reshape if possible
                    if embedding_array.size == query_image_array.size:
                        embedding_array = embedding_array.reshape(query_image_array.shape)
                
                try:
                    image_similarity = float(np.dot(query_image_array, embedding_array) / 
                                           (np.linalg.norm(query_image_array) * np.linalg.norm(embedding_array)))
                except Exception as e:
                    print(f"Error computing image similarity: {e}")
                    image_similarity = 0.0
                
                # If only image query, use image similarity as combined similarity
                if query_text_embedding is None:
                    combined_similarity = image_similarity
            
            # If both text and image queries are provided, compute weighted combination
            if query_text_embedding is not None and query_image_embedding is not None:
                combined_similarity = (image_weight * image_similarity) + ((1 - image_weight) * text_similarity)
            
            results.append({
                'image_path': img_path,
                'similarity': float(combined_similarity),
                'image_similarity': float(image_similarity) if query_image_embedding is not None else None,
                'text_similarity': float(text_similarity) if query_text_embedding is not None else None
            })
        
        # Sort by combined similarity (descending) and get top 10
        results.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = results[:10]
        
        # Format the response to match what the frontend expects
        formatted_images = []
        for result in top_results:
            formatted_images.append({
                'url': result['image_path'],
                'similarity': result['similarity']
            })
        
        return jsonify({
            'success': True,
            'formatted_images': formatted_images,
            'similar_images': formatted_images  # Providing both formats for compatibility
        })
        
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