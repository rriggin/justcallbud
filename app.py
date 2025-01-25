from flask import Flask, render_template, request, jsonify
from datetime import datetime
import os
import logging
import requests

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Use Modal in production, local Ollama in development
USE_MODAL = os.getenv('FLASK_ENV') == 'production'

if USE_MODAL:
    from modal import App
    modal_app = App.lookup("just-call-bud-prod")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
async def chat():
    try:
        user_message = request.form.get('content', '')
        prompt = f"""You are Bud, a friendly and knowledgeable AI assistant...
        User: {user_message}
        Assistant: """
        
        if USE_MODAL:
            # Production: Use Modal
            response = await modal_app.get_llama_response.remote(prompt)
            content = response
        else:
            # Development: Use local Ollama
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
        return jsonify({
            'content': f"Error: {str(e)}",
            'isUser': False,
            'timestamp': datetime.now().isoformat()
        }), 500

port = int(os.environ.get('PORT', 5001))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)