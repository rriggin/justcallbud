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
from langchain_openai import ChatOpenAI

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
app.secret_key = os.getenv('FLASK_SECRET_KEY')

# Use Modal unless explicitly in development
USE_MODAL = os.getenv('FLASK_ENV') != 'development'
logger.info(f"USE_MODAL: {USE_MODAL}")
logger.info(f"Running in {'Production' if USE_MODAL else 'Development'} mode")
logger.info(f"Environment: {'Production' if USE_MODAL else 'Development'}")

# Initialize Modal globally
modal_function = None
modal_initialized = False

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)
logger.info(f"Supabase initialized with URL: {os.getenv('SUPABASE_URL')}")

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

# Initialize LLM based on environment
if USE_MODAL:
    llm = ChatOpenAI()  # Will need to be replaced with Modal's function
else:
    llm = ChatOllama(model="llama2")

# Initialize Redis client
redis_client = Redis(host='localhost', port=6379, db=0, decode_responses=True)
CACHE_EXPIRATION = 3600  # Cache for 1 hour

def get_cache_key(conversation_id, prompt_text, history):
    """Generate a unique cache key based on conversation context"""
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

@app.route('/')
def home():
    # Clear the conversation_id from session to force a new conversation on page load
    if 'conversation_id' in session:
        del session['conversation_id']
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Ensure we have a valid conversation
        conversation_id = get_or_create_conversation()
        
        prompt_text = request.form.get('content', '')
        if not prompt_text:
            raise ValueError("Message content cannot be empty")
            
        logger.info(f"Received prompt: {prompt_text}")
        
        # Get conversation history
        history = get_conversation_history(conversation_id)
        
        # Check cache first
        cache_key = get_cache_key(conversation_id, prompt_text, history)
        cached_response = redis_client.get(cache_key)
        
        if cached_response:
            logger.info("Cache hit! Using cached response")
            response_text = cached_response
        else:
            logger.info("Cache miss. Generating new response")
            # Get response using LangChain
            if USE_MODAL:
                response_text = modal_function.remote(prompt_text)
            else:
                # Create the chain properly
                chain = prompt | llm
                response = chain.invoke({
                    "history": history,
                    "input": prompt_text
                })
                response_text = response.content

            if not response_text:
                raise ValueError("Empty response from AI model")
                
            # Cache the response
            redis_client.setex(cache_key, CACHE_EXPIRATION, response_text)

        # Get the current max message number for this conversation
        result = supabase.table('messages') \
            .select('message_number') \
            .eq('conversation_id', conversation_id) \
            .order('message_number', desc=True) \
            .limit(1) \
            .execute()
        
        current_max_number = 0
        if result.data:
            current_max_number = result.data[0].get('message_number', 0)

        # Store user message
        message_data = {
            'content': prompt_text,
            'is_user': True,
            'conversation_id': conversation_id,
            'message_number': current_max_number + 1,
            'created_at': datetime.now().isoformat()
        }
        supabase.table('messages').insert(message_data).execute()

        # Store bot response
        bot_message_data = {
            'content': response_text,
            'is_user': False,
            'conversation_id': conversation_id,
            'message_number': current_max_number + 2,
            'created_at': datetime.now().isoformat()
        }
        supabase.table('messages').insert(bot_message_data).execute()
            
        return jsonify({
            "content": response_text,
            "isUser": False,
            "timestamp": datetime.now().isoformat()
        })
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            "content": f"Error: {str(e)}",
            "isUser": False,
            "timestamp": datetime.now().isoformat()
        }), 400
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

port = int(os.environ.get('PORT', 5001))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)