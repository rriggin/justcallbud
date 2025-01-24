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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
async def chat():
    try:
        user_message = request.form.get('content', '')
        if not user_message:
            raise ValueError("No message content provided")

        # Create the prompt
        prompt = f"""You are Bud, a friendly and knowledgeable AI assistant specializing in home maintenance and repairs.
        
        User: {user_message}
        Assistant: """

        # Call Modal function
        response = await modal_app.get_llama_response.remote(prompt)
        
        return jsonify({
            'content': response,
            'isUser': False,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
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

port = int(os.environ.get('PORT', 5001))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)