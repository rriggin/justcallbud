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
        logger.info("=== Starting Modal Initialization ===")
        app = modal.App.lookup("just-call-bud-prod")
        logger.info(f"Found app: {app}")
        
        # Get the function directly
        get_llama = getattr(app, 'get_llama_response', None)
        if get_llama:
            logger.info("Found get_llama_response function")
            modal_initialized = True
            return get_llama
        else:
            logger.error("Function not found on app")
            return None
            
    except Exception as e:
        logger.error(f"Modal initialization error: {str(e)}")
        return None

# Initialize Modal when app starts
init_modal()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
async def chat():
    try:
        user_message = request.form.get('content', '')
        logger.info(f"Received message: {user_message[:100]}...")

        prompt = f"""You are Bud, a friendly and knowledgeable AI assistant...
        User: {user_message}
        Assistant: """
        
        if USE_MODAL:
            logger.info("Using Modal in production...")
            try:
                # Make sure Modal is properly initialized
                if not hasattr(modal_app, 'get_llama_response'):
                    logger.error("Modal function not found")
                    raise Exception("Modal not properly initialized")
                
                # Call Modal function
                response = await modal_app.get_llama_response.remote(prompt)
                content = response
                logger.info(f"Modal response received: {content[:100]}...")
            except Exception as modal_error:
                logger.error(f"Modal error: {str(modal_error)}", exc_info=True)
                raise
        else:
            logger.info("Using local Ollama...")
            # Use asyncio for local requests
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: requests.post(
                'http://localhost:11434/api/generate',
                json={"model": "llama2", "prompt": prompt, "stream": False}
            ).json())
            content = response['response']

        return jsonify({
            'content': content,
            'isUser': False,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        return jsonify({
            'content': f"Error: {str(e)}",
            'isUser': False,
            'timestamp': datetime.now().isoformat()
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