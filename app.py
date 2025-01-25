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
modal_function = None
modal_initialized = False

def init_modal():
    global modal_function, modal_initialized
    try:
        logger.info("=== Starting Modal Initialization ===")
        modal_function = modal.Function.lookup("just-call-bud-prod", "get_llama_response")
        if modal_function:
            modal_initialized = True
            logger.info("Modal function found and initialized successfully")
    except Exception as e:
        logger.error(f"Modal initialization error: {str(e)}")
        logger.error(f"Full error details: {repr(e)}")

# Force initialization at startup
logger.info("=== Forcing Modal Initialization ===")
init_modal()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    if not modal_initialized:
        logger.error("Modal not initialized")
        return jsonify({"error": "Modal not initialized"}), 500
        
    try:
        prompt = request.form.get('content', '')
        logger.info(f"Received prompt: {prompt}")
        
        logger.info("Calling Modal function...")
        response = modal_function.remote(prompt).result()
        logger.info(f"Modal response received: {response}")
        
        if not response:
            logger.error("Empty response from Modal")
            return jsonify({"error": "Empty response from AI"}), 500
            
        return jsonify({
            "content": response,
            "isUser": False,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        logger.error(f"Full error details: {repr(e)}")
        return jsonify({
            "content": f"Error: {str(e)}",
            "isUser": False,
            "timestamp": datetime.now().isoformat()
        }), 500

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