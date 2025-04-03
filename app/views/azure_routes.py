import os
import pickle
import json
import logging
import numpy as np
from flask import Blueprint, request
from app.services.azure_service import AzureService
from app.utils.helpers import save_uploaded_file, get_file_url, create_cors_response
from app.utils.s3_helper import upload_file_to_s3

logger = logging.getLogger(__name__)
azure_bp = Blueprint('azure', __name__)
azure_service = AzureService()

# Define possible file paths for embeddings
embeddings_paths = [
    os.environ.get("AZURE_EMBEDDINGS_FILE", "static/json/azure_embeddings_s3_images.pkl"),
    os.path.join('static', 'json', 'azure_embeddings_s3_images.pkl'),
    os.path.join(os.getcwd(), 'static', 'json', 'azure_embeddings_s3_images.pkl'),
    os.path.join('static', 'json', 'azure_embeddings_s3_images.pkl'),
    os.path.join(os.getcwd(), 'static', 'json', 'azure_embeddings_s3_images.pkl'),
]

# Try to find and load the embeddings file
precomputed_data = None
for path in embeddings_paths:
    try:
        if os.path.exists(path):
            logger.info(f"Found embeddings file at: {path}")
            
            # Load based on file extension
            if path.endswith('.pkl'):
                with open(path, 'rb') as f:
                    precomputed_data = pickle.load(f)
            elif path.endswith('.json'):
                with open(path, 'r') as f:
                    precomputed_data = json.load(f)
            
            # Verify the data structure
            if precomputed_data and 'embeddings' in precomputed_data and 'image_urls' in precomputed_data:
                logger.info(f"Successfully loaded embeddings for {len(precomputed_data['image_urls'])} images")
                break
            else:
                logger.warning(f"File {path} doesn't have the expected structure")
                precomputed_data = None
    except Exception as e:
        logger.error(f"Error loading embeddings from {path}: {str(e)}")
        precomputed_data = None

if precomputed_data is None:
    logger.error(f"Failed to load embeddings from any of the attempted paths: {embeddings_paths}")

@azure_bp.route('/process', methods=['POST', 'OPTIONS'])
def process_image():
    """Process an image with Azure Vision API"""
    if request.method == 'OPTIONS':
        return create_cors_response()
        
    try:
        if 'image' not in request.files:
            logger.error("No image file in request")
            return create_cors_response({"error": "No image file provided"}, 400)
            
        image_file = request.files['image']
        if not image_file.filename:
            logger.error("Empty filename in request")
            return create_cors_response({"error": "Empty filename"}, 400)
            
        # Save the file and get its URL
        file_path = save_uploaded_file(image_file)
        image_url = get_file_url(file_path)
        
        logger.info(f"Processing image: {image_file.filename}")
        
        # Get image embedding
        image_embedding = azure_service.vectorize_image(image_url)
        if image_embedding is None:
            logger.error("Failed to generate image embedding")
            return create_cors_response({"error": "Failed to generate image embedding"}, 500)
            
        # Return the embedding
        return create_cors_response({
            "success": True,
            "image_url": image_url,
            "embedding_size": len(image_embedding),
            "embedding": image_embedding[:10] + ["..."]  # Return truncated embedding for display
        })
        
    except Exception as e:
        logger.exception(f"Error processing image with Azure: {str(e)}")
        return create_cors_response({"error": str(e)}, 500)

@azure_bp.route('/search', methods=['POST', 'OPTIONS'])
def search_images():
    """Search for similar images using Azure Vision API"""
    if request.method == 'OPTIONS':
        return create_cors_response()
        
    try:
        if not precomputed_data:
            logger.error("Precomputed embeddings not available")
            return create_cors_response({"error": "Precomputed embeddings not available"}, 500)
            
        # Get parameters from form data or JSON
        image_url = None
        image_embedding = None
        
        # Check if we have form data with an image
        if 'image' in request.files and request.files['image'].filename:
            image_file = request.files['image']
            logger.info(f"Received image file: {image_file.filename}")
            
            # Save the file locally first
            file_path = save_uploaded_file(image_file)
            
            # Upload to S3 to get a publicly accessible URL
            s3_url = upload_file_to_s3(file_path)
            
            if s3_url:
                image_url = s3_url
                logger.info(f"Image uploaded to S3: {image_url}")
                
                # Get image embedding using the S3 URL
                image_embedding = azure_service.vectorize_image(image_url)
                if image_embedding is None:
                    logger.warning("Failed to generate image embedding from S3 URL")
            else:
                logger.error("Failed to upload image to S3")
        
        # Get text from form data or JSON
        query_text = ''
        if request.form and 'text' in request.form:
            query_text = request.form.get('text', '')
        elif request.is_json and request.json:
            query_text = request.json.get('query_text', '')
        
        # Get other parameters with safe fallbacks
        try:
            top_k = int(request.form.get('top_k', 7) if request.form else 
                        (request.json.get('top_k', 7) if request.is_json and request.json else 7))
        except (ValueError, TypeError):
            top_k = 7
            
        try:
            image_weight = float(request.form.get('image_weight', 0.4) if request.form else 
                               (request.json.get('image_weight', 0.4) if request.is_json and request.json else 0.4))
        except (ValueError, TypeError):
            image_weight = 0.4
            
        try:
            text_weight = float(request.form.get('text_weight', 0.6) if request.form else 
                              (request.json.get('text_weight', 0.6) if request.is_json and request.json else 0.6))
        except (ValueError, TypeError):
            text_weight = 0.6
        
        logger.info(f"Search parameters: text='{query_text}', top_k={top_k}, image_weight={image_weight}, text_weight={text_weight}")
        
        if not image_url and not query_text:
            logger.error("Neither image nor query text provided")
            return create_cors_response({"error": "Please provide either an image or query text"}, 400)
            
        # Get text embedding if we have query text
        text_embedding = None
        if query_text:
            logger.info(f"Generating embedding for text: {query_text}")
            text_embedding = azure_service.vectorize_text(query_text)
            if text_embedding is None:
                logger.warning("Failed to generate text embedding")
                
        if image_embedding is None and text_embedding is None:
            logger.error("Failed to generate both image and text embeddings")
            return create_cors_response({"error": "Failed to generate embeddings"}, 500)
            
        # Combine embeddings
        combined_embedding = azure_service.combine_embeddings(
            image_embedding, 
            text_embedding,
            image_weight,
            text_weight
        )
        
        # Convert precomputed embeddings to numpy arrays
        reference_embeddings = [np.array(emb) for emb in precomputed_data['embeddings']]
        reference_urls = precomputed_data['image_urls']
        
        # Find similar images
        result_urls, similarities = azure_service.find_similar_images(
            combined_embedding,
            reference_embeddings,
            reference_urls,
            top_k
        )
        
        # Format results for frontend
        similar_images = []
        for i, (url, similarity) in enumerate(zip(result_urls, similarities)):
            similar_images.append({
                "rank": i + 1,
                "url": url,
                "similarity": float(similarity)
            })
            
        # Create response in the format expected by frontend
        return create_cors_response({
            "success": True,
            "message": f"Found {len(similar_images)} similar images using Azure Vision API",
            "similar_images": similar_images,
            "query": {
                "image_url": image_url,
                "text": query_text,
                "image_weight": image_weight,
                "text_weight": text_weight
            }
        })
        
    except Exception as e:
        logger.exception(f"Error searching images with Azure: {str(e)}")
        return create_cors_response({
            "success": False,
            "message": f"Error searching images: {str(e)}"
        }, 500)

@azure_bp.route('/vectorize_text', methods=['POST', 'OPTIONS'])
def vectorize_text():
    """Generate vector embedding for text"""
    if request.method == 'OPTIONS':
        return create_cors_response()
        
    try:
        data = request.json or {}
        text = data.get('text', '')
        
        if not text:
            logger.error("No text provided")
            return create_cors_response({"error": "No text provided"}, 400)
            
        embedding = azure_service.vectorize_text(text)
        if embedding is None:
            logger.error("Failed to generate text embedding")
            return create_cors_response({"error": "Failed to generate text embedding"}, 500)
            
        return create_cors_response({
            "success": True,
            "text": text,
            "embedding_size": len(embedding),
            "embedding": embedding[:10] + ["..."]  # Return truncated embedding for display
        })
        
    except Exception as e:
        logger.exception(f"Error vectorizing text with Azure: {str(e)}")
        return create_cors_response({"error": str(e)}, 500) 