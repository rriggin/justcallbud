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
volume = modal.Volume(path="/cache", size=20)

def create_image():
    return (
        modal.Image.debian_slim()
        .pip_install([
            "torch==2.1.0",
            "transformers==4.37.2",
            "accelerate>=0.26.1",
            "huggingface-hub>=0.19.4",
            "safetensors>=0.4.1"
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

@app.cls(
    image=create_image(),
    gpu="A10G",  # Using A10G for good performance/cost ratio
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": volume},
    container_idle_timeout=300  # Keep container alive for 5 minutes between requests
)
class LLM:
    def __enter__(self):
        logger.info("Initializing LLM class...")
        
        # Set PyTorch to use CUDA
        if torch.cuda.is_available():
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
            self.device = torch.device("cuda")
        else:
            logger.warning("CUDA not available. Using CPU.")
            self.device = torch.device("cpu")
            
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            cache_dir="/cache",
            use_fast=True  # Use faster tokenizer implementation
        )
        
        # Load model with optimizations
        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            cache_dir="/cache",
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_cache=True  # Enable KV cache for faster inference
        )
        
        # Move model to GPU and optimize
        self.model.to(self.device)
        if hasattr(self.model, 'eval'):
            self.model.eval()  # Set to evaluation mode
            
        # Set up optimized pipeline
        logger.info("Setting up pipeline...")
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
        
        logger.info("Model initialization complete")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup is handled by Modal
        pass

    @torch.inference_mode()  # More efficient than no_grad for inference
    def generate(self, prompt_text: str, history: List[Any]) -> str:
        logger.info("Generating response...")
        
        formatted_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt_text}\n\nBud:"
        
        with torch.cuda.amp.autocast():  # Enable automatic mixed precision
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
    volumes={"/cache": volume}
)
async def chat(data: dict) -> str:
    logger.info("Chat function called")
    try:
        prompt_text = data.get("content") or data.get("prompt_text", "")
        raw_history = data.get("history", [])
        
        # Convert history format
        history = []
        for msg in raw_history:
            if isinstance(msg, str):
                history.append(HumanMessage(content=msg))
            else:
                content = msg.get("content", "")
                msg_type = msg.get("type", "human")
                if msg_type == "human":
                    history.append(HumanMessage(content=content))
                else:
                    history.append(AIMessage(content=content))
        
        with LLM() as llm:
            response = llm.generate(prompt_text, history)
            return str(response)
            
    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    app.serve() 