from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import logging
import os
from werkzeug.utils import secure_filename
from app.views.twelvelabs_routes import twelvelabs_bp
from app import create_app
import dotenv

dotenv.load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Application Configuration
class Config:
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB limit
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize Flask app
app = create_app()

# Configure CORS properly
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "https://67e7c87e9879df0035944485--peaceful-bonbon-e63768.netlify.app"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Create upload directory if it doesn't exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Helper functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def handle_options_request():
    """Handle CORS preflight requests"""
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    return response

def save_uploaded_file(file):
    """Save an uploaded file to the configured upload folder"""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return file_path
    return None

# Routes
@app.route('/vertex/chat', methods=['POST', 'OPTIONS'])
def vertex_chat():
    """
    Handle chat requests with optional image uploads
    """
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return handle_options_request()
    
    logger.info("Request received at /vertex/chat")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request headers: {dict(request.headers)}")
    
    try:
        # Get text from form data
        text = request.form.get('text', '')
        logger.info(f"Received text: {text}")
        
        # Process image if present
        image_path = None
        if 'image' in request.files:
            image = request.files['image']
            if image.filename:
                logger.info(f"Received image: {image.filename}")
                image_path = save_uploaded_file(image)
                logger.info(f"Saved image to: {image_path}")
        
        # Here you would typically process the text and image
        # For example, send to a model, database, etc.
        
        # Create response
        response_data = {
            'message': f'Successfully processed your request. Text: "{text}"',
            'image_received': image_path is not None,
            'image_path': image_path
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    """Simple endpoint to test if the server is running"""
    return jsonify({'message': 'Server is working!'})

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    return jsonify({'error': 'File too large'}), 413

# Add this line with the other blueprint registrations
app.register_blueprint(twelvelabs_bp, url_prefix='/twelvelabs')

# Run the application
if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True, port=5000)