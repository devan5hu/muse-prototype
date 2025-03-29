import os

class Config:
    """Application configuration"""
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB limit
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    DEBUG = os.environ.get('FLASK_DEBUG', 'True') == 'True'
    PORT = int(os.environ.get('FLASK_PORT', 5000)) 