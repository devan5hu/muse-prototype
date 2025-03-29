from flask import Blueprint
from app.controllers.azure_controller import handle_azure_request

azure_bp = Blueprint('azure', __name__)

@azure_bp.route('/azure', methods=['POST', 'OPTIONS'])
def azure():
    """Route for Azure AI endpoint"""
    return handle_azure_request() 