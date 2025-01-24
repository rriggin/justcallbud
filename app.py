from gevent import monkey
monkey.patch_all()

from flask import Flask, render_template, request, jsonify
from datetime import datetime
import os
import logging
from modal import App

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get the deployed Modal app
modal_app = App("just-call-bud-prod")

messages = []  # Store messages in memory

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
async def chat():
    try:
        logger.info("=== New Chat Request ===")
        user_message = request.form.get('content', '')
        logger.info(f"User message: {user_message[:100]}...")  # Log first 100 chars

        if not user_message:
            logger.error("Empty message received")
            raise ValueError("No message content provided")

        # Create the prompt
        prompt = f"""You are Bud, a friendly and knowledgeable AI assistant specializing in home maintenance and repairs.
        
        User: {user_message}
        Assistant: """
        
        logger.info("Calling Modal function...")
        try:
            response = await modal_app.get_llama_response.remote(prompt)
            logger.info(f"Modal response received: {response[:100]}...")  # Log first 100 chars
        except Exception as modal_error:
            logger.error(f"Modal error: {str(modal_error)}", exc_info=True)
            raise

        return jsonify({
            'content': response,
            'isUser': False,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        return jsonify({
            'content': f"I apologize, but I'm having trouble processing your request right now. Error: {str(e)}",
            'isUser': False,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health')
def health():
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/messages', methods=['GET'])
def get_messages():
    return jsonify(messages if messages else [])

@app.route('/api/messages', methods=['DELETE'])
def clear_messages():
    global messages
    messages = []
    return jsonify({'status': 'success'})

port = int(os.environ.get('PORT', 5001))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)