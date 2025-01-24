from gevent import monkey
monkey.patch_all()

from flask import Flask, render_template, request, jsonify, url_for
from datetime import datetime
import os
import requests
import subprocess
import time
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import logging
from modal import App

# Set up logging first
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

# Now we can use logger
MODAL_TOKEN_ID = os.getenv('MODAL_TOKEN_ID')
MODAL_TOKEN_SECRET = os.getenv('MODAL_TOKEN_SECRET')

if not MODAL_TOKEN_ID or not MODAL_TOKEN_SECRET:
    logger.warning("Modal tokens not found in environment variables. Some features may not work.")
else:
    logger.info("Modal tokens found in environment")

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def ensure_ollama_running():
    # Skip Ollama check in production
    if os.getenv('FLASK_ENV') == 'production':
        logger.info("Skipping Ollama check in production (using Modal)")
        return True
        
    # Original Ollama check code for local development
    try:
        logger.info("Checking if Ollama is running...")
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code != 200:
            logger.error(f"Ollama responded with status code: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return False
        logger.info("✅ Ollama is running and responding correctly")
        return True
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Connection error when checking Ollama: {str(e)}")
        logger.info("Attempting to start Ollama...")
        try:
            process = subprocess.Popen(['ollama', 'serve'], 
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
            logger.info(f"Started Ollama process with PID: {process.pid}")
            
            # Wait for Ollama to start
            for i in range(10):
                try:
                    time.sleep(1)
                    logger.info(f"Attempt {i+1}: Checking if Ollama is responding...")
                    response = requests.get('http://localhost:11434/api/tags')
                    if response.status_code == 200:
                        logger.info("✅ Ollama started successfully")
                        return True
                except requests.exceptions.ConnectionError:
                    continue
            logger.error("❌ Failed to start Ollama after 10 attempts")
            return False
        except FileNotFoundError:
            logger.error("❌ Error: Ollama is not installed or not in PATH")
            return False

# Start Ollama when app starts
ensure_ollama_running()

# Initialize empty messages array on server start
messages = []

# Get the deployed app
modal_app = App("just-call-bud-prod")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    global messages
    messages = []  # Clear messages on home page load
    return render_template('index.html')

@app.route('/api/messages', methods=['GET'])
def get_messages():
    return jsonify(messages if messages else [])

# Replace Llama initialization with Ollama API call
def get_llama_response(prompt):
    logger.info("\n" + "="*50)
    logger.info("Starting Llama request...")
    try:
        # Now try to generate response
        logger.info(f"Sending prompt to Ollama: {prompt[:100]}...")  # Log first 100 chars of prompt
        response = requests.post('http://localhost:11434/api/generate', 
            json={
                "model": "llama2",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": 2048,
                    "num_thread": 8,
                    "temperature": 0.7,
                    "top_k": 20,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }, timeout=60)
        
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        logger.info(f"Raw response content: {response.text}")  # Log full response

        if response.status_code != 200:
            logger.error(f"Error from Ollama: {response.text}")
            raise Exception(f"Ollama returned status code {response.status_code}")

        try:
            response_json = response.json()
            logger.info(f"Parsed JSON response: {response_json}")
        except Exception as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            raise

        if 'response' not in response_json:
            logger.error(f"Unexpected response format: {response_json}")
            raise Exception("No 'response' field in Ollama response")

        logger.info("Successfully got response from Ollama")
        logger.info(f"AI Response: {response_json['response'][:100]}...")  # Log first 100 chars of response
        return response_json['response']

    except Exception as e:
        logger.error(f"Error in get_llama_response: {str(e)}", exc_info=True)
        raise

@app.route('/api/chat', methods=['POST'])
async def chat():
    try:
        logger.info("\n" + "="*50)
        logger.info("🔄 Chat endpoint called")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request form data: {dict(request.form)}")

        if not ensure_ollama_running():
            logger.error("❌ Ollama not running")
            return jsonify({
                'content': "Sorry, the AI service is not available right now. Please make sure Ollama is installed and running.",
                'isUser': False,
                'timestamp': datetime.now().isoformat()
            }), 503

        user_message = request.form.get('content', '')
        if not user_message:
            logger.error("No content in request")
            raise ValueError("No message content provided")

        logger.info(f"Processing message: {user_message}")

        # Create the prompt
        prompt = f"""You are Bud, a friendly and knowledgeable AI assistant specializing in home maintenance and repairs.
        
        User: {user_message}
        Assistant: """
        
        logger.info(f"Created prompt: {prompt}")

        # Store user message
        user_msg = {
            'content': user_message,
            'isUser': True,
            'timestamp': datetime.now().isoformat()
        }
        messages.append(user_msg)
        logger.info("Stored user message")

        # Call the deployed Modal function
        response = await modal_app.get_llama_response.remote(prompt)
        logger.info("Successfully got AI response")

        # Create response
        ai_msg = {
            'content': response,
            'isUser': False,
            'timestamp': datetime.now().isoformat()
        }
        messages.append(ai_msg)
        logger.info("Stored AI response")
        
        logger.info("Sending response back to client")
        return jsonify(ai_msg)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        logger.error("Full traceback:", exc_info=True)
        error_msg = {
            'content': f"I apologize, but I'm having trouble processing your request right now. Error: {str(e)}",
            'isUser': False,
            'timestamp': datetime.now().isoformat()
        }
        messages.append(error_msg)
        return jsonify(error_msg), 500  # Added 500 status code

@app.route('/api/messages', methods=['DELETE'])
def clear_messages():
    global messages
    messages = []
    return jsonify({'status': 'success'})

@app.route('/test', methods=['POST'])
def test():
    logger.info("Test endpoint called!")
    logger.info(f"Form data: {dict(request.form)}")
    return jsonify({"status": "ok"})

@app.route('/health')
def health():
    try:
        logger.debug("Health check called")
        # Test Modal connection
        with modal_app.run():
            response = modal_app.test.remote()
        logger.debug(f"Modal response: {response}")
        return jsonify({
            'status': 'healthy',
            'modal': 'connected',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Add port from environment for Render
port = int(os.environ.get('PORT', 5001))

if __name__ == '__main__':
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY not found in environment variables")
    app.run(host='0.0.0.0', port=port)