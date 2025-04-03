import os
import boto3
import logging
from botocore.exceptions import ClientError
from uuid import uuid4

logger = logging.getLogger(__name__)

def upload_file_to_s3(file_path, bucket_name=None, object_name=None):
    """
    Upload a file to an S3 bucket and return its public URL
    
    Args:
        file_path (str): Path to the file to upload
        bucket_name (str): Name of the bucket to upload to
        object_name (str): S3 object name (if None, file_name will be used)
        
    Returns:
        str: Public URL of the uploaded file or None if upload fails
    """
    # Get AWS credentials from environment variables
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    aws_region = os.environ.get('AWS_REGION', 'us-east-1')
    
    if not aws_access_key or not aws_secret_key:
        logger.error("AWS credentials not found in environment variables")
        return None
        
    # Use default bucket name if not provided - use muse-prototype instead of muse-objects-1
    if bucket_name is None:
        bucket_name = os.environ.get('AWS_S3_BUCKET', 'muse-objects-1')
        
    # Generate a unique object name if not provided
    if object_name is None:
        file_name = os.path.basename(file_path)
        # Add a UUID to ensure uniqueness
        object_name = f"image_uploads/{uuid4().hex}_{file_name}"
    elif not object_name.startswith('image_uploads/'):
        # Ensure the object is in the image_uploads folder
        object_name = f"image_uploads/{object_name}"
    
    logger.info(f"Attempting to upload {file_path} to {bucket_name}/{object_name}")
    
    # Create S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    
    try:
        # Try uploading without ACL first (in case user doesn't have ACL permissions)
        try:
            s3_client.upload_file(
                file_path, 
                bucket_name, 
                object_name
            )
        except ClientError as e:
            if 'AccessDenied' in str(e):
                logger.warning("Access denied with default upload, trying without ACL")
                # If that fails, try with public-read ACL
                s3_client.upload_file(
                    file_path, 
                    bucket_name, 
                    object_name,
                    ExtraArgs={'ACL': 'public-read'}
                )
        
        # Generate the URL
        url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{object_name}"
        logger.info(f"File uploaded successfully to S3: {url}")
        return url
    except ClientError as e:
        logger.error(f"Error uploading file to S3: {str(e)}")

def get_s3_url_without_params(url):
    """
    Convert a pre-signed S3 URL to a regular S3 URL by removing query parameters
    
    Args:
        url (str): The S3 URL, possibly with query parameters
        
    Returns:
        str: The S3 URL without query parameters
    """
    if not url:
        return url
        
    # Remove query parameters (everything after ?)
    if '?' in url:
        return url.split('?')[0]
    return url

def generate_presigned_url(bucket_name, object_name, expiration=3600):
    """
    Generate a presigned URL for an S3 object
    
    Args:
        bucket_name (str): Name of the bucket
        object_name (str): Name of the object
        expiration (int): Time in seconds for the URL to remain valid
        
    Returns:
        str: Presigned URL
    """
    # Get AWS credentials from environment variables
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    aws_region = os.environ.get('AWS_REGION', 'us-east-1')
    
    if not aws_access_key or not aws_secret_key:
        logger.error("AWS credentials not found in environment variables")
        return None
    
    # Create S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    
    try:
        # Generate the presigned URL
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
        return response
    except ClientError as e:
        logger.error(f"Error generating presigned URL: {str(e)}")
        return None