from flask import Blueprint, jsonify, send_from_directory
from app.controllers.test_controller import handle_test_request
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

test_bp = Blueprint('test', __name__)

@test_bp.route('/test', methods=['GET'])
def test():
    """Route for test endpoint"""
    return handle_test_request()

@test_bp.route('/images')
def test_images():
    """Test route to list available images"""
    images_dir = os.path.join(app.static_folder, 'images')
    if os.path.exists(images_dir):
        images = os.listdir(images_dir)
        image_urls = [f'/static/images/{img}' for img in images if img.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        return jsonify({
            'count': len(image_urls),
            'images': image_urls[:20]  # Show first 20 images
        })
    else:
        return jsonify({
            'error': f'Images directory not found: {images_dir}',
            'static_folder': app.static_folder
        })

@test_bp.route('/check-images')
def check_images():
    """Check where images are located in the static folder"""
    static_folder = app.static_folder
    
    # Check various possible image directories
    image_dirs = [
        'images',
        'selected_images',
        '.'  # Root of static folder
    ]
    
    results = {}
    
    for dir_name in image_dirs:
        dir_path = os.path.join(static_folder, dir_name)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            files = os.listdir(dir_path)
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            results[dir_name] = {
                'path': dir_path,
                'exists': True,
                'image_count': len(image_files),
                'sample_images': image_files[:5] if image_files else []
            }
        else:
            results[dir_name] = {
                'path': dir_path,
                'exists': False
            }
    
    # Also check the pickle file to see what paths it contains
    try:
        from app.services.similarity_service import EMBEDDINGS_PICKLE_PATH, load_embeddings
        
        if os.path.exists(EMBEDDINGS_PICKLE_PATH):
            embeddings_data = load_embeddings()
            
            # Extract some sample file paths
            sample_paths = []
            if isinstance(embeddings_data, dict):
                if 'file_paths' in embeddings_data:
                    sample_paths = embeddings_data['file_paths'][:5]
                else:
                    sample_paths = list(embeddings_data.keys())[:5]
            elif isinstance(embeddings_data, list):
                if all(isinstance(item, tuple) and len(item) == 2 for item in embeddings_data[:5]):
                    sample_paths = [item[0] for item in embeddings_data[:5]]
                elif all(isinstance(item, dict) and 'file_path' in item for item in embeddings_data[:5] if isinstance(item, dict)):
                    sample_paths = [item['file_path'] for item in embeddings_data[:5]]
            
            results['pickle_file'] = {
                'path': EMBEDDINGS_PICKLE_PATH,
                'exists': True,
                'sample_paths': sample_paths
            }
        else:
            results['pickle_file'] = {
                'path': EMBEDDINGS_PICKLE_PATH,
                'exists': False
            }
    except Exception as e:
        results['pickle_file'] = {
            'path': EMBEDDINGS_PICKLE_PATH if 'EMBEDDINGS_PICKLE_PATH' in locals() else 'unknown',
            'error': str(e)
        }
    
    return jsonify(results)

@test_bp.route('/check-all-images')
def check_all_images():
    """Check all image directories and list files"""
    import os
    from flask import current_app, jsonify
    
    static_folder = current_app.static_folder
    all_images_dir = os.path.join(static_folder, 'all_images')
    
    # Create the directory if it doesn't exist
    os.makedirs(all_images_dir, exist_ok=True)
    
    # Check if directory exists
    if not os.path.exists(all_images_dir):
        return jsonify({
            'error': f'Directory not found: {all_images_dir}',
            'static_folder': static_folder
        })
    
    # List all files in the directory
    files = os.listdir(all_images_dir)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    # Check for specific files mentioned in the logs
    missing_files = ['image_579.jpg', 'image_524.jpg', 'image_999.jpg', 'image_250.jpg']
    file_status = {}
    for file in missing_files:
        file_status[file] = file in files
    
    return jsonify({
        'directory': all_images_dir,
        'exists': True,
        'total_files': len(files),
        'image_files': len(image_files),
        'sample_files': files[:10] if files else [],
        'specific_files_exist': file_status
    })

@test_bp.route('/serve-image/<filename>')
def serve_image_direct(filename):
    """Directly serve an image from the all_images directory"""
    from flask import send_from_directory, current_app
    import os
    
    # Add .jpg extension if not present
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
        filename = f"{filename}.jpg"
    
    all_images_dir = os.path.join(current_app.static_folder, 'all_images')
    
    # Check if file exists
    full_path = os.path.join(all_images_dir, filename)
    if os.path.exists(full_path):
        return send_from_directory(all_images_dir, filename)
    else:
        return f"Image not found: {full_path}", 404 

@test_bp.route('/image-test')
def image_test_page():
    """Render a test page with images"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Test</title>
        <style>
            .image-container { display: flex; flex-wrap: wrap; }
            .image-item { margin: 10px; text-align: center; }
            img { max-width: 200px; max-height: 200px; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>Image Test Page</h1>
        
        <h2>Direct Static URLs</h2>
        <div class="image-container">
            <div class="image-item">
                <img src="/static/all_images/image_579.jpg" alt="Image 579">
                <p>image_579.jpg</p>
            </div>
            <div class="image-item">
                <img src="/static/all_images/image_524.jpg" alt="Image 524">
                <p>image_524.jpg</p>
            </div>
            <div class="image-item">
                <img src="/static/all_images/image_999.jpg" alt="Image 999">
                <p>image_999.jpg</p>
            </div>
        </div>
        
        <h2>Using Serve Route</h2>
        <div class="image-container">
            <div class="image-item">
                <img src="/serve-image/image_579" alt="Image 579">
                <p>image_579 (serve route)</p>
            </div>
            <div class="image-item">
                <img src="/serve-image/image_524" alt="Image 524">
                <p>image_524 (serve route)</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html 

@test_bp.route('/check-pickle')
def check_pickle():
    """Check the contents of the embeddings pickle file"""
    from app.services.similarity_service import EMBEDDINGS_PICKLE_PATH, load_embeddings
    import os
    
    if not os.path.exists(EMBEDDINGS_PICKLE_PATH):
        return jsonify({
            'error': f'Pickle file not found: {EMBEDDINGS_PICKLE_PATH}'
        })
    
    embeddings_data = load_embeddings()
    
    # Extract sample file paths
    sample_paths = []
    if isinstance(embeddings_data, dict):
        if 'file_paths' in embeddings_data:
            sample_paths = embeddings_data['file_paths'][:10]
        else:
            sample_paths = list(embeddings_data.keys())[:10]
    elif isinstance(embeddings_data, list):
        if all(isinstance(item, tuple) and len(item) == 2 for item in embeddings_data[:10]):
            sample_paths = [item[0] for item in embeddings_data[:10]]
        elif all(isinstance(item, dict) and 'file_path' in item for item in embeddings_data[:10] if isinstance(item, dict)):
            sample_paths = [item['file_path'] for item in embeddings_data[:10]]
    
    return jsonify({
        'pickle_file': EMBEDDINGS_PICKLE_PATH,
        'exists': True,
        'sample_paths': sample_paths,
        'data_type': str(type(embeddings_data))
    })

@test_bp.route('/embedding-images')
def embedding_images():
    """Display images from the embeddings file"""
    from app.services.similarity_service import load_embeddings
    
    embeddings_data = load_embeddings()
    
    # Extract file paths
    file_paths = []
    if isinstance(embeddings_data, dict):
        if 'file_paths' in embeddings_data:
            file_paths = embeddings_data['file_paths'][:20]
        else:
            file_paths = list(embeddings_data.keys())[:20]
    elif isinstance(embeddings_data, list):
        if all(isinstance(item, tuple) and len(item) == 2 for item in embeddings_data[:20]):
            file_paths = [item[0] for item in embeddings_data[:20]]
        elif all(isinstance(item, dict) and 'file_path' in item for item in embeddings_data[:20] if isinstance(item, dict)):
            file_paths = [item['file_path'] for item in embeddings_data[:20]]
    
    # Generate HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Embedding Images</title>
        <style>
            .image-container { display: flex; flex-wrap: wrap; }
            .image-item { margin: 10px; text-align: center; }
            img { max-width: 200px; max-height: 200px; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>Images from Embeddings</h1>
        
        <div class="image-container">
    """
    
    for path in file_paths:
        filename = os.path.basename(path)
        html += f"""
            <div class="image-item">
                <img src="/static/all_images/{filename}" alt="{filename}">
                <p>{filename}</p>
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html 

@test_bp.route('/compare-embeddings')
def compare_embeddings():
    """Compare two embeddings to test cosine similarity"""
    from app.services.similarity_service import load_embeddings
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    embeddings_data = load_embeddings()
    
    # Extract embeddings
    stored_embeddings = []
    file_paths = []
    
    if isinstance(embeddings_data, dict):
        if 'embeddings' in embeddings_data and 'file_paths' in embeddings_data:
            stored_embeddings = embeddings_data['embeddings']
            file_paths = embeddings_data['file_paths']
        else:
            file_paths = list(embeddings_data.keys())
            stored_embeddings = list(embeddings_data.values())
    elif isinstance(embeddings_data, list):
        if all(isinstance(item, tuple) and len(item) == 2 for item in embeddings_data):
            file_paths = [item[0] for item in embeddings_data]
            stored_embeddings = [item[1] for item in embeddings_data]
        elif all(isinstance(item, dict) and 'embedding' in item and 'file_path' in item 
                for item in embeddings_data if isinstance(item, dict)):
            file_paths = [item['file_path'] for item in embeddings_data]
            stored_embeddings = [item['embedding'] for item in embeddings_data]
    
    # Process embeddings if they are dictionaries
    processed_embeddings = []
    for emb in stored_embeddings:
        if isinstance(emb, dict) and 'embedding' in emb:
            processed_embeddings.append(emb['embedding'])
        elif isinstance(emb, list) or isinstance(emb, np.ndarray):
            processed_embeddings.append(emb)
    
    if len(processed_embeddings) < 2:
        return jsonify({'error': 'Not enough embeddings to compare'})
    
    # Compare first two embeddings
    emb1 = np.array(processed_embeddings[0], dtype=np.float32).reshape(1, -1)
    emb2 = np.array(processed_embeddings[1], dtype=np.float32).reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(emb1, emb2)[0][0]
    
    # Calculate Euclidean distance
    euclidean_distance = np.linalg.norm(emb1 - emb2)
    
    # Check if embeddings are normalized
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    return jsonify({
        'file1': file_paths[0] if len(file_paths) > 0 else 'unknown',
        'file2': file_paths[1] if len(file_paths) > 1 else 'unknown',
        'cosine_similarity': float(similarity),
        'euclidean_distance': float(euclidean_distance),
        'norm1': float(norm1),
        'norm2': float(norm2),
        'embedding1_sample': processed_embeddings[0][:5] if len(processed_embeddings[0]) > 5 else processed_embeddings[0],
        'embedding2_sample': processed_embeddings[1][:5] if len(processed_embeddings[1]) > 5 else processed_embeddings[1],
        'embedding1_shape': emb1.shape,
        'embedding2_shape': emb2.shape
    }) 

@test_bp.route('/check-json')
def check_json():
    """Check the contents of the embeddings JSON file"""
    from app.services.similarity_service import EMBEDDINGS_JSON_PATH
    import os
    import json
    
    if not os.path.exists(EMBEDDINGS_JSON_PATH):
        return jsonify({
            'error': f'JSON file not found: {EMBEDDINGS_JSON_PATH}'
        })
    
    try:
        with open(EMBEDDINGS_JSON_PATH, 'r') as f:
            embeddings_data = json.load(f)
        
        # Get a sample of the data
        sample_keys = list(embeddings_data.keys())[:5]
        sample_data = {k: embeddings_data[k] for k in sample_keys}
        
        # Check structure
        structure_info = {}
        if sample_keys:
            first_key = sample_keys[0]
            first_item = embeddings_data[first_key]
            if isinstance(first_item, dict):
                structure_info['keys_in_item'] = list(first_item.keys())
                
                if 'embedding' in first_item:
                    structure_info['embedding_length'] = len(first_item['embedding'])
                    structure_info['embedding_sample'] = first_item['embedding'][:5]
        
        return jsonify({
            'file_path': EMBEDDINGS_JSON_PATH,
            'exists': True,
            'total_entries': len(embeddings_data),
            'sample_keys': sample_keys,
            'sample_data': sample_data,
            'structure_info': structure_info
        })
    except Exception as e:
        return jsonify({
            'error': f'Error reading JSON file: {str(e)}'
        }) 