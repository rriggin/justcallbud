from flask import Flask, render_template, request, jsonify
import asyncio
from datetime import datetime
import os
import logging
import requests
import sys
import time
import modal

# Enhanced logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Use Modal in production, local Ollama in development
USE_MODAL = os.getenv('FLASK_ENV') == 'production'
logger.info(f"Environment: {'Production' if USE_MODAL else 'Development'}")

# Initialize Modal globally
modal_app = None
modal_initialized = False

def init_modal():
    global modal_app, modal_initialized
    try:
        modal_app = modal.App.lookup("just-call-bud-prod")
        if modal_app:
            modal_initialized = True
            logger.info("Modal initialized successfully")
    except Exception as e:
        logger.error(f"Modal initialization error: {str(e)}")

# Initialize Modal when app starts
init_modal()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
async def chat():
    if not modal_initialized:
        return jsonify({"error": "Modal not initialized"}), 500
        
    try:
        prompt = request.form.get('content', '')
        response = await modal_app.get_llama_response.remote(prompt)
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['POST'])
def test():
    try:
        logger.info("Test endpoint called")
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"Test endpoint error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/messages', methods=['GET'])
def get_messages():
    return jsonify([])  # Return empty array for now

@app.route('/api/messages', methods=['DELETE'])
def clear_messages():
    return jsonify({'status': 'success'})

@app.route('/health')
def health():
    try:
        if USE_MODAL and not modal_initialized:
            raise Exception("Modal not properly initialized")
            
        return jsonify({
            'status': 'healthy',
            'environment': 'production' if USE_MODAL else 'development',
            'modal_status': 'initialized' if modal_initialized else 'not initialized',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'modal_status': 'not initialized',
            'timestamp': datetime.now().isoformat()
        }), 500

port = int(os.environ.get('PORT', 5001))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)