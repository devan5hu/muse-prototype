import os
import time
import numpy as np
import requests
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class AzureService:
    def __init__(self):
        self.vision_key = os.environ.get("AZURE_VISION_KEY")
        self.vision_endpoint = os.environ.get("AZURE_VISION_ENDPOINT")
        self.api_version = "2024-02-01"
        self.model_version = "2023-04-15"
        
        if not self.vision_key or not self.vision_endpoint:
            logger.error("Azure Vision API credentials not found in environment variables")
            raise ValueError("Azure Vision API credentials not configured")
        
        logger.info("Azure Vision service initialized")

    def vectorize_text(self, text: str) -> Optional[List[float]]:
        """Generate vector embedding for text using Azure Vision API"""
        if not isinstance(text, str) or not text.strip():
            logger.warning("Skipping empty text vectorization")
            return None
            
        logger.info(f"Vectorizing text: {text[:50]}...")
        url = f"{self.vision_endpoint}/computervision/retrieval:vectorizeText?api-version={self.api_version}&model-version={self.model_version}"
        headers = {"Content-Type": "application/json", "Ocp-Apim-Subscription-Key": self.vision_key}
        data = {"text": text.strip()}
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                logger.info("Text vectorization successful")
                return result.get("vector")
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Text vectorization attempt {attempt+1} failed, retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to vectorize text after {max_retries} attempts: {str(e)}")
                    return None

    def vectorize_image(self, image_url: str) -> Optional[List[float]]:
        """Generate vector embedding for image using Azure Vision API"""
        if not isinstance(image_url, str) or not image_url.startswith("http"):
            logger.error(f"Invalid image URL: {image_url}")
            return None
            
        url = f"{self.vision_endpoint}/computervision/retrieval:vectorizeImage?api-version={self.api_version}&model-version={self.model_version}"
        headers = {"Content-Type": "application/json", "Ocp-Apim-Subscription-Key": self.vision_key}
        payload = {"url": image_url}
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                logger.info(f"Vectorizing image URL: {image_url}")
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                logger.info("Image vectorization successful")
                return result.get("vector")
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    error_details = e.response.text if hasattr(e, 'response') and hasattr(e.response, 'text') else str(e)
                    logger.warning(f"Image vectorization attempt {attempt+1} failed, retrying in {wait_time}s: {error_details}")
                    time.sleep(wait_time)
                else:
                    error_details = e.response.text if hasattr(e, 'response') and hasattr(e.response, 'text') else str(e)
                    logger.error(f"Failed to vectorize image after {max_retries} attempts: {error_details}")
                    return None

    def normalize_vector(self, vector: List[float]) -> Optional[np.ndarray]:
        """Normalize a vector to unit length"""
        if vector is None:
            return None
        vector_np = np.array(vector)
        norm = np.linalg.norm(vector_np)
        return vector_np if norm == 0 else vector_np / norm

    def cosine_similarity(self, vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
        """Calculate cosine similarity between two vectors"""
        if vec1 is None or vec2 is None:
            return 0.0
        vec1_norm = self.normalize_vector(vec1)
        vec2_norm = self.normalize_vector(vec2)
        return float(np.dot(vec1_norm, vec2_norm))

    def combine_embeddings(self, 
                          image_embedding: Optional[List[float]], 
                          text_embedding: Optional[List[float]], 
                          image_weight: float = 0.4, 
                          text_weight: float = 0.6) -> Optional[np.ndarray]:
        """Combine image and text embeddings with specified weights"""
        if image_embedding is None and text_embedding is None:
            logger.warning("Both image and text embeddings are None, cannot combine")
            return None
            
        if image_embedding is None:
            logger.info("Using only text embedding (image embedding is None)")
            return self.normalize_vector(text_embedding)
            
        if text_embedding is None:
            logger.info("Using only image embedding (text embedding is None)")
            return self.normalize_vector(image_embedding)
            
        norm_image_emb = self.normalize_vector(np.array(image_embedding))
        norm_text_emb = self.normalize_vector(np.array(text_embedding))
        combined = (image_weight * norm_image_emb) + (text_weight * norm_text_emb)
        logger.info(f"Combined embeddings with weights: image={image_weight}, text={text_weight}")
        return self.normalize_vector(combined)

    def find_similar_images(self, 
                           combined_embedding: np.ndarray, 
                           reference_embeddings: List[np.ndarray], 
                           reference_urls: List[str], 
                           top_k: int = 10) -> Tuple[List[str], List[float]]:
        """Find top-k similar images based on embedding similarity"""
        if combined_embedding is None:
            logger.error("Cannot find similar images: combined embedding is None")
            return [], []
            
        if len(reference_embeddings) != len(reference_urls):
            logger.error(f"Mismatch between embeddings ({len(reference_embeddings)}) and URLs ({len(reference_urls)})")
            return [], []
            
        logger.info(f"Finding top {top_k} similar images from {len(reference_embeddings)} reference images")
        similarities = [self.cosine_similarity(combined_embedding, emb) for emb in reference_embeddings]
        top_indices = np.argsort(similarities)[-top_k:][::-1]  # Get indices of top k similarities
        
        result_urls = [reference_urls[i] for i in top_indices]
        result_similarities = [similarities[i] for i in top_indices]
        
        logger.info(f"Found {len(result_urls)} similar images")
        return result_urls, result_similarities 