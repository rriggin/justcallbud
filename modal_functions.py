import modal
from modal import Image, Secret
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import torch
import logging
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create app and volume for persistent storage
app = modal.App("just-call-bud-prod")

# Create a volume for model weights
MODEL_DIR = "/model"
volume = modal.Volume.from_name("llama-weights", create_if_missing=True)

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
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            cache_dir=MODEL_DIR,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_cache=True,
            token=self.hf_token
        )
        
        # Move model to GPU and optimize
        self.model.to(self.device)
        self.model.eval()
            
        # Set up pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
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
    volumes={MODEL_DIR: volume},
    container_idle_timeout=300
)
async def chat(data: dict) -> str:
    """Handle chat requests"""
    logger.info("Chat function called")
    try:
        prompt_text = data.get("content") or data.get("prompt_text", "")
        if not prompt_text:
            raise ValueError("No prompt text provided")
            
        # Initialize and load model
        model = Model()
        model.load()
        
        # Generate response
        response = model.generate(prompt_text)
        return str(response)
            
    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    app.serve() 