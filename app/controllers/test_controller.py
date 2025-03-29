from flask import jsonify

def handle_test_request():
    """Simple endpoint to test if the server is running"""
    return jsonify({'message': 'Server is working!'}) 