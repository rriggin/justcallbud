import modal
from modal import Image, Secret, Volume
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import torch
import logging
import os
import json
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create app and volumes for persistent storage
app = modal.App("just-call-bud-prod")

# Create volumes for model weights and response cache
MODEL_DIR = "/model"
CACHE_DIR = "/cache"
model_volume = modal.Volume.from_name("llama-weights", create_if_missing=True)
cache_volume = modal.Volume.from_name("response-cache", create_if_missing=True)

def get_cache_key(prompt_text: str, image_path: str = None) -> str:
    """Generate a unique cache key based on input"""
    context = f"{prompt_text}:{image_path if image_path else ''}"
    return f"response_{hashlib.md5(context.encode()).hexdigest()}.json"

def save_to_cache(cache_key: str, response: str):
    """Save response to cache file"""
    try:
        cache_path = os.path.join(CACHE_DIR, cache_key)
        with open(cache_path, 'w') as f:
            json.dump({'response': response}, f)
        logger.info(f"Response cached successfully at {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to cache response: {str(e)}")

def load_from_cache(cache_key: str) -> str | None:
    """Load response from cache file if it exists"""
    try:
        cache_path = os.path.join(CACHE_DIR, cache_key)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Cache hit! Loading from {cache_path}")
                return data['response']
    except Exception as e:
        logger.warning(f"Failed to load from cache: {str(e)}")
    return None

def create_image():
    return (
        modal.Image.debian_slim()
        .pip_install([
            "torch==2.1.0",
            "transformers==4.37.2",
            "accelerate>=0.26.1",
            "huggingface-hub>=0.19.4",
            "safetensors>=0.4.1",
            "langchain>=0.1.0",
            "langchain-core>=0.1.30",
            "langchain-community>=0.0.27"
        ])
        .run_commands(
            # Install CUDA dependencies
            "pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html"
        )
    )

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

class Model:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.device = None
        # Get Hugging Face token from environment
        self.hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            logger.warning("HUGGINGFACE_TOKEN not found in environment, trying HF_TOKEN")
            self.hf_token = os.environ.get("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("No Hugging Face token found in environment variables")

    def load(self):
        """Load the model and tokenizer"""
        logger.info("Loading model and tokenizer...")
        
        # Set PyTorch to use CUDA
        if torch.cuda.is_available():
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
            self.device = torch.device("cuda")
        else:
            logger.warning("CUDA not available. Using CPU.")
            self.device = torch.device("cpu")
            
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            cache_dir=MODEL_DIR,
            use_fast=True,
            token=self.hf_token
        )
        
        # Load model with optimizations
        logger.info("Loading model with device_map='auto' for optimal placement...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            cache_dir=MODEL_DIR,
            device_map="auto",  # This handles device placement automatically
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_cache=True,
            token=self.hf_token
        )
        
        # Model is already placed on devices by device_map="auto"
        self.model.eval()
            
        # Set up pipeline with the model's device mapping
        logger.info("Setting up pipeline...")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",  # Use the same device mapping as the model
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            torch_dtype=torch.float16,
            use_cache=True
        )
        
        logger.info("Model loaded successfully")

    @torch.inference_mode()
    def generate(self, prompt_text: str) -> str:
        """Generate a response for the given prompt"""
        logger.info("Generating response...")
        
        formatted_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt_text}\n\nBud:"
        
        with torch.cuda.amp.autocast():
            result = self.pipe(
                formatted_prompt,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )[0]['generated_text']
        
        return result.strip()

@app.function(
    image=create_image(),
    gpu="A10G",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        MODEL_DIR: model_volume,
        CACHE_DIR: cache_volume
    },
    container_idle_timeout=300
)
async def chat(data: dict) -> str:
    """Handle chat requests"""
    logger.info("Chat function called")
    try:
        prompt_text = data.get("content") or data.get("prompt_text", "")
        image_path = data.get("image_path")
        
        if not prompt_text:
            raise ValueError("No prompt text provided")

        # Try to get cached response
        cache_key = get_cache_key(prompt_text, image_path)
        cached_response = load_from_cache(cache_key)
        if cached_response:
            return cached_response
        
        logger.info("Cache miss. Generating new response...")
            
        # If there's an image, add context to the prompt
        if image_path:
            prompt_text = f"The user has uploaded an image ({image_path}). Please analyze this image along with their message: {prompt_text}"
            
        # Initialize and load model
        model = Model()
        model.load()
        
        # Generate response
        response = model.generate(prompt_text)
        response_text = str(response)

        # Cache the response
        save_to_cache(cache_key, response_text)

        return response_text
            
    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    app.serve() 