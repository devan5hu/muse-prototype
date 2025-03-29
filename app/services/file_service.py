import os
import logging
from werkzeug.utils import secure_filename
from flask import current_app

logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file):
    """Save an uploaded file to the upload folder"""
    try:
        if file and file.filename:
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
                
                # Create upload folder if it doesn't exist
                os.makedirs(upload_folder, exist_ok=True)
                
                file_path = os.path.join(upload_folder, filename)
                file.save(file_path)
                
                logger.info(f"Saved uploaded file to {file_path}")
                return file_path
            else:
                logger.warning(f"File type not allowed: {file.filename}")
                raise ValueError(f"File type not allowed: {file.filename}")
        else:
            logger.warning("No file provided")
            raise ValueError("No file provided")
            
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}", exc_info=True)
        raise 