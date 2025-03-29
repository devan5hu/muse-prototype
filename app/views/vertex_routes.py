from flask import Blueprint
from app.controllers.vertex_controller import handle_vertex_request

chat_bp = Blueprint('vertex', __name__)

@chat_bp.route('/vertex/chat', methods=['POST', 'OPTIONS'])
def vertex_chat():
    """Route for Vertex AI embedding endpoint"""
    return handle_vertex_request() 