from flask import Flask
from flask_cors import CORS
import os
import logging
from app.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_app(config_class=Config):
    """Application factory function"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Configure CORS - we'll use only one method for consistency
    # Option 1: Use Flask-CORS extension (recommended for most cases)
    CORS(app, resources={
        r"/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000", 
                       "https://peaceful-bonbon-e63768.netlify.app", 
                       "https://*.netlify.app"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }
    })
    
    # Option 2: Use after_request handler (comment out Option 1 if using this)
    # @app.after_request
    # def apply_cors_headers(response):
    #     response.headers.add('Access-Control-Allow-Origin', 'https://peaceful-bonbon-e63768.netlify.app')
    #     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    #     response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    #     response.headers.add('Access-Control-Allow-Credentials', 'true')
    #     return response
    
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Preload AI models
    with app.app_context():
        logger.info("Preloading AI models...")
        # Import here to avoid circular imports
        from app.services.vertex_service import initialize_vertex_ai
        initialize_vertex_ai()
        logger.info("AI models preloaded")
    
    # Register blueprints
    from app.views.test_routes import test_bp
    from app.views.titan_routes import titan_bp
    from app.views.twelvelabs_routes import twelvelabs_bp
    from app.views.azure_routes import azure_bp
    from app.views.cohere_routes import cohere_bp
    from app.views.voyage_routes import voyage_bp
    
    app.register_blueprint(test_bp)
    app.register_blueprint(titan_bp)
    app.register_blueprint(twelvelabs_bp, url_prefix='/twelvelabs')
    app.register_blueprint(azure_bp)
    app.register_blueprint(cohere_bp)
    app.register_blueprint(voyage_bp)
    
    # Register error handlers
    from app.utils.helpers import handle_404_error, handle_413_error
    app.register_error_handler(404, handle_404_error)
    app.register_error_handler(413, handle_413_error)
    
    logger.info(f"Registering blueprint: twelvelabs_bp with prefix: /twelvelabs")
    logger.info(f"Registered routes: {[str(rule) for rule in app.url_map.iter_rules()]}")
    
    logger.info("Application initialized successfully")
    return app 