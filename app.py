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
from dotenv import load_dotenv
from supabase import create_client, Client
import re
from email_validator import validate_email, EmailNotValidError

# Load environment variables from .env
load_dotenv()

# Enhanced logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Log environment for debugging
logger.info(f"FLASK_ENV: {os.getenv('FLASK_ENV')}")

app = Flask(__name__)
# Generate a random secret key if not provided
if not os.environ.get('FLASK_SECRET_KEY'):
    logger.warning("FLASK_SECRET_KEY not set, using random key")
app.secret_key = 'dev-key-please-change'  # Simple for now

# Use Modal unless explicitly in development
USE_MODAL = os.getenv('FLASK_ENV') != 'development'
logger.info(f"USE_MODAL: {USE_MODAL}")
logger.info(f"Running in {'Production' if USE_MODAL else 'Development'} mode")
logger.info(f"Environment: {'Production' if USE_MODAL else 'Development'}")

# Initialize Modal globally
modal_function = None
modal_initialized = False

# Store conversations in memory
conversation_store = {}

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)
logger.info(f"Supabase initialized with URL: {os.getenv('SUPABASE_URL')}")

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
if USE_MODAL:
    init_modal()
else:
    logger.info("Development mode - skipping Modal initialization")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        if not USE_MODAL:
            # Use Ollama in development
            prompt = request.form.get('content', '')
            formatted_prompt = f"""<s>[INST] <<SYS>>
                You are Bud, a friendly and knowledgeable AI handyman assistant. 
                Core rules:
                1. Never use asterisks (*) or describe actions
                2. Never use emotes or roleplay elements
                3. Maintain a professional, direct tone
                4. Focus only on home repair advice
                5. Start responses with "Hello" or direct answers
                6. Avoid casual expressions or emojis
                
                Your purpose is to provide practical, safety-focused home maintenance advice.
                Keep all responses focused on technical details and solutions.
                <</SYS>>
                
                {prompt} [/INST]"""
            
            response = requests.post('http://localhost:11434/api/generate',
                json={
                    "model": "llama2",
                    "prompt": formatted_prompt,
                    "stream": False
                }
            ).json()
            
            response_text = response.get('response', '')
            
            return jsonify({
                "content": response_text,
                "isUser": False,
                "timestamp": datetime.now().isoformat()
            })
        
        conversation_id = session.get('conversation_id')
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            session['conversation_id'] = conversation_id
            conversation_store[conversation_id] = []
        
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

def validate_phone(phone):
    # Remove any non-digit characters
    phone = re.sub(r'\D', '', phone)
    # Check if it's a valid US phone number (10 digits)
    if len(phone) != 10:
        raise ValueError("Phone number must be 10 digits")
    return phone

def validate_zip(zip_code):
    # Check if it's a valid US ZIP code (5 digits)
    if not re.match(r'^\d{5}$', zip_code):
        raise ValueError("ZIP code must be 5 digits")
    return zip_code

@app.route('/api/quote-request', methods=['POST'])
def quote_request():
    try:
        logger.info("=== Starting Quote Request ===")
        # Get form data
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        zip_code = request.form.get('zip', '').strip()
        
        logger.info(f"Received form data: name={name}, email={email}, phone={phone}, zip={zip_code}")
        
        # Validate data
        if not name:
            raise ValueError("Name is required")
        
        # Validate email
        try:
            valid = validate_email(email)
            email = valid.email
        except EmailNotValidError as e:
            raise ValueError(f"Invalid email: {str(e)}")
        
        # Validate phone
        phone = validate_phone(phone)
        
        # Validate ZIP
        zip_code = validate_zip(zip_code)
        
        # Insert into Supabase
        data = {
            'name': name,
            'email': email,
            'phone': phone,
            'zip_code': zip_code,
            'status': 'new',
            'created_at': 'now()'
        }
        
        logger.info(f"Attempting to insert data: {data}")
        result = supabase.table('quote_requests').insert(data).execute()
        response_data = result.model_dump() if hasattr(result, 'model_dump') else result.dict()
        logger.info(f"Insert result: {response_data}")
        
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Full error details: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/test-supabase', methods=['GET'])
def test_supabase():
    try:
        # Test data
        test_data = {
            'name': 'Test User',
            'email': 'test@example.com',
            'phone': '1234567890',
            'zip_code': '12345',
            'status': 'test',
        }
        
        logger.info(f"Testing Supabase insert with data: {test_data}")
        
        # Try to insert
        result = supabase.table('quote_requests').insert(test_data).execute()
        
        # Convert result to dict
        response_data = result.model_dump() if hasattr(result, 'model_dump') else result.dict()
        logger.info(f"Test insert result: {response_data}")
        return jsonify({
            'status': 'success',
            'result': response_data
        })
        
    except Exception as e:
        logger.error(f"Supabase test error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

port = int(os.environ.get('PORT', 5001))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)