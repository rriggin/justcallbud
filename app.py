from flask import Flask, render_template, request, jsonify
from datetime import datetime
import os
import logging
import requests
import sys

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

if USE_MODAL:
    try:
        from modal import App
        modal_app = App.lookup("just-call-bud-prod")
        logger.info("Modal initialized successfully")
    except Exception as e:
        logger.error(f"Modal initialization error: {str(e)}", exc_info=True)

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
                response = await modal_app.get_llama_response.remote(prompt)
                content = response
                logger.info(f"Modal response received: {content[:100]}...")
            except Exception as modal_error:
                logger.error(f"Modal error: {str(modal_error)}", exc_info=True)
                raise
        else:
            logger.info("Using local Ollama...")
            response = requests.post('http://localhost:11434/api/generate', 
                json={"model": "llama2", "prompt": prompt, "stream": False}
            ).json()
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

port = int(os.environ.get('PORT', 5001))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)