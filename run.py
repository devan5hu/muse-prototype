from app import create_app
import logging

logger = logging.getLogger(__name__)
app = create_app()

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=app.config['DEBUG'], port=app.config['PORT']) 