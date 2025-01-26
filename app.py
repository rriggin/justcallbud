from flask import Flask, render_template, request, jsonify, session
import asyncio
from datetime import datetime
import os
import logging
import requests
import sys
import time
import modal
import uuid

# Enhanced logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Generate a random secret key if not provided
if not os.environ.get('FLASK_SECRET_KEY'):
    logger.warning("FLASK_SECRET_KEY not set, using random key")
app.secret_key = 'dev-key-please-change'  # Simple for now

# Use Modal in production, local Ollama in development
USE_MODAL = os.getenv('FLASK_ENV') == 'production'
logger.info(f"Environment: {'Production' if USE_MODAL else 'Development'}")

# Initialize Modal globally
modal_function = None
modal_initialized = False

# Store conversations in memory
conversation_store = {}

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
    conversation_id = session.get('conversation_id')
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        session['conversation_id'] = conversation_id
        conversation_store[conversation_id] = []
    
    try:
        prompt = request.form.get('content', '')
        logger.info(f"Received prompt: {prompt}")
        
        response = modal_function.remote(prompt)
        
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

@app.route('/api/model-info')
def model_info():
    return jsonify({
        "models": [
            {
                "name": "meta-llama/Llama-2-7b-chat-hf",
                "model": "Llama-2-7b-chat-hf",
                "details": {
                    "family": "llama",
                    "parameter_size": "7B",
                    "provider": "Meta/Hugging Face",
                    "deployment": "Modal GPU (A10G)",
                    "quantization": "float16"  # From our torch_dtype setting
                }
            }
        ]
    })

port = int(os.environ.get('PORT', 5001))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)