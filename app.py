from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
import asyncio
from datetime import datetime
import os
import logging
import requests
import sys
import time
import modal
import uuid
import json
import hashlib
from redis import Redis
from dotenv import load_dotenv
from supabase import create_client, Client
import re
from email_validator import validate_email, EmailNotValidError
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama
from werkzeug.utils import secure_filename

# Load environment variables from .env
load_dotenv()

# Enhanced logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Initialize Flask app with secret key first
app = Flask(__name__)

# Set the secret key directly in the config
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-only-secret-key-do-not-use-in-production')
logger.info("Flask config initialized")
logger.info(f"Secret key is set: {'yes' if 'SECRET_KEY' in app.config else 'no'}")

# Log environment for debugging
logger.info(f"FLASK_ENV: {os.getenv('FLASK_ENV')}")

# Use Modal unless explicitly in development
USE_MODAL = os.getenv('FLASK_ENV') != 'development'
logger.info(f"USE_MODAL: {USE_MODAL}")
logger.info(f"Running in {'Production' if USE_MODAL else 'Development'} mode")
logger.info(f"Environment: {'Production' if USE_MODAL else 'Development'}")

# Initialize Modal globally
modal_function = None
modal_initialized = False
pending_jobs = {}

# Initialize Modal function
if USE_MODAL:
    try:
        logger.info("Initializing Modal client...")
        modal_function = modal.Function.from_name("just-call-bud-prod", "chat")
        modal_initialized = True
        logger.info("Modal function initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Modal client: {str(e)}")
        logger.error("Full error details:", exc_info=True)
else:
    logger.info("Running in development mode, using local Ollama")
    llm = ChatOllama(model="llama2")
    logger.info("Local Ollama initialized for development")

# Initialize Supabase client based on environment
if USE_MODAL:  # Production
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    logger.info("Using production Supabase database")
else:  # Development
    supabase_url = os.getenv('SUPABASE_DEV_URL', os.getenv('SUPABASE_URL'))
    supabase_key = os.getenv('SUPABASE_DEV_KEY', os.getenv('SUPABASE_KEY'))
    logger.info("Using development Supabase database")

supabase: Client = create_client(supabase_url, supabase_key)
logger.info(f"Supabase initialized with URL: {supabase_url}")

# Initialize LangChain components
SYSTEM_PROMPT = """You are Bud, a friendly and knowledgeable AI handyman assistant. 
Core rules:
1. Never use asterisks (*) or describe actions
2. Never use emotes or roleplay elements
3. Maintain a professional, direct tone
4. Focus only on home repair advice
5. Start responses with "Hello" or direct answers
6. Avoid casual expressions or emojis

Your purpose is to provide practical, safety-focused home maintenance advice.
Keep all responses focused on technical details and solutions."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Redis initialization code - commented out as we're using Modal volumes for caching
# If we need distributed caching across services in the future, we can re-enable this
"""
try:
    redis_url = os.getenv('REDIS_URL')  # Render provides this environment variable
    if redis_url and USE_MODAL:  # In production
        redis_client = Redis.from_url(redis_url, decode_responses=True)
        redis_enabled = True
        logger.info("Redis cache initialized using REDIS_URL in production")
    elif not USE_MODAL:  # In development
        try:
            redis_client = Redis(host='localhost', port=6379, db=0, decode_responses=True)
            redis_enabled = True
            logger.info("Redis cache initialized using localhost for development")
        except Exception as e:
            logger.warning(f"Local Redis not available in development: {str(e)}")
            redis_client = None
            redis_enabled = False
    else:
        logger.warning("Redis URL not found in production environment")
        redis_client = None
        redis_enabled = False
except Exception as e:
    redis_client = None
    redis_enabled = False
    logger.warning(f"Redis cache not available - caching disabled: {str(e)}")
"""

# Redis-related variables - commented out as not currently in use
"""
CACHE_EXPIRATION = 3600  # Cache for 1 hour

def get_cache_key(conversation_id, prompt_text, history):
    if not redis_enabled:
        return None
        
    history_str = json.dumps([{
        'content': msg.content,
        'is_user': isinstance(msg, HumanMessage)
    } for msg in history])
    
    context = f"{conversation_id}:{prompt_text}:{history_str}"
    return f"chat_response:{hashlib.md5(context.encode()).hexdigest()}"
"""

def get_cache_key(conversation_id, prompt_text, history):
    """Generate a unique cache key based on conversation context"""
    if not redis_enabled:
        return None
        
    # Convert history to a string representation
    history_str = json.dumps([{
        'content': msg.content,
        'is_user': isinstance(msg, HumanMessage)
    } for msg in history])
    
    # Create a unique hash of the conversation context
    context = f"{conversation_id}:{prompt_text}:{history_str}"
    return f"chat_response:{hashlib.md5(context.encode()).hexdigest()}"

def get_or_create_anonymous_user():
    """Get existing or create new anonymous user session"""
    if 'user_id' not in session:
        try:
            # Generate a more standard anonymous email format
            random_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
            anonymous_email = f"anonymous.{random_id}@justcallbud.com"
            
            # Sign up anonymously with Supabase
            auth_response = supabase.auth.sign_up({
                "email": anonymous_email,
                "password": uuid.uuid4().hex
            })
            session['user_id'] = auth_response.user.id
        except Exception as e:
            logger.error(f"Error creating anonymous user: {str(e)}")
            raise
    return session['user_id']

def create_new_conversation(user_id):
    """Create a new conversation in Supabase"""
    try:
        # Create conversation with only required fields
        result = supabase.table('conversations').insert({
            'user_id': user_id,
            'created_at': datetime.now().isoformat()
        }).execute()
        conversation = result.data[0]
        session['conversation_id'] = conversation['id']
        logger.info(f"Created new conversation: {conversation['id']}")
        return conversation['id']
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise

def get_or_create_conversation():
    """Get existing or create new conversation"""
    user_id = get_or_create_anonymous_user()
    # Always create a new conversation
    return create_new_conversation(user_id)

def get_conversation_history(conversation_id):
    """Get conversation history from Supabase and format for LangChain"""
    messages = supabase.table('messages') \
        .select('*') \
        .eq('conversation_id', conversation_id) \
        .order('message_number', desc=False) \
        .execute()
    
    history = []
    for msg in messages.data:
        if msg['is_user']:
            history.append(HumanMessage(content=msg['content']))
        else:
            history.append(AIMessage(content=msg['content']))
    
    return history

# Add this after other configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    # Clear the conversation_id from session to force a new conversation on page load
    if 'conversation_id' in session:
        del session['conversation_id']
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Get form data
        prompt_text = request.form.get('content', '')
        if not prompt_text:
            return jsonify({"error": "No content provided"}), 400

        # Handle image upload if present
        image_path = None
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Create timestamp-based unique filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_path = f"/static/uploads/{filename}"
                # Append image info to prompt
                prompt_text = f"{prompt_text}\n[Image uploaded: {image_path}]"
                logger.info(f"Image saved at: {filepath}")

        # Convert history format for the model
        history = []  # We'll implement history later if needed
        
        if USE_MODAL:
            try:
                if not modal_initialized or not modal_function:
                    raise RuntimeError("Modal function not properly initialized")
                    
                logger.info("Calling Modal function...")
                response_text = modal_function.remote({
                    "prompt_text": prompt_text,
                    "history": history,
                    "image_path": image_path
                })
                logger.info("Modal function call completed")
                
                # Return JSON response
                return jsonify({
                    "content": response_text,
                    "isUser": False,
                    "image_path": image_path
                })
                
            except Exception as e:
                error_msg = f"Error in Modal operation: {str(e)}"
                logger.error(error_msg)
                logger.error("Full error details:", exc_info=True)
                return jsonify({"error": error_msg}), 500
        else:
            # Local LLM path (unchanged)
            llm = ChatOllama(model="llama2")
            response = llm.generate(prompt_text, history)
            return jsonify({
                "content": response,
                "isUser": False,
                "image_path": image_path
            })
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error("Full error details:", exc_info=True)
        return jsonify({"error": str(e)}), 500

def stream_cached_response(response):
    """Stream a cached response to maintain consistent behavior"""
    yield f"data: {json.dumps({'content': response, 'done': False})}\n\n"
    yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"

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
    try:
        conversation_id = get_or_create_conversation()
        messages = supabase.table('messages') \
            .select('*') \
            .eq('conversation_id', conversation_id) \
            .order('message_number', desc=False) \
            .execute()
        return jsonify([{
            'content': msg['content'],
            'isUser': msg['is_user'],
            'timestamp': msg['created_at']
        } for msg in messages.data])
    except Exception as e:
        logger.error(f"Error getting messages: {str(e)}")
        return jsonify([])

@app.route('/api/messages', methods=['DELETE'])
def clear_messages():
    try:
        # Create new conversation
        user_id = get_or_create_anonymous_user()
        create_new_conversation(user_id)
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error clearing messages: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

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

@app.route('/api/test-redis', methods=['GET'])
def test_redis():
    try:
        if not redis_enabled or not redis_client:
            return jsonify({
                'status': 'error',
                'message': 'Redis is not enabled or client is not initialized'
            }), 500

        # Test basic Redis operations
        test_key = 'test_key'
        test_value = 'test_value'
        
        # Set a value
        redis_client.set(test_key, test_value, ex=60)  # expires in 60 seconds
        
        # Get the value back
        retrieved_value = redis_client.get(test_key)
        
        # Delete the test key
        redis_client.delete(test_key)
        
        return jsonify({
            'status': 'success',
            'connection': 'established',
            'operations': {
                'set': 'success',
                'get': 'success',
                'retrieved_value': retrieved_value,
                'delete': 'success'
            },
            'redis_url': os.getenv('REDIS_URL', 'not_set')[:20] + '...'  # Only show beginning of URL for security
        })
        
    except Exception as e:
        logger.error(f"Redis test error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'redis_enabled': redis_enabled,
            'redis_url_exists': bool(os.getenv('REDIS_URL'))
        }), 500

port = int(os.environ.get('PORT', 5001))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)