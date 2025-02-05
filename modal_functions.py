import modal
from modal import Image, Secret, NetworkFileSystem
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging
import os
from huggingface_hub import login
from typing import List, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a persistent storage for model weights
CACHE_DIR = "/root/model_cache"
stub = modal.Stub("just-call-bud-prod")
cache = NetworkFileSystem.persisted("llama-cache")

def create_image():
    return (
        modal.Image.debian_slim()
        .pip_install([
            "torch==2.1.0",
            "transformers==4.37.2",
            "langchain>=0.1.0",
            "langchain_core>=0.1.30",
            "accelerate>=0.26.1",
            "huggingface-hub>=0.19.4",
            "einops>=0.7.0",
            "safetensors>=0.4.1",
            "fastapi[standard]>=0.109.0"
        ])
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

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

@stub.cls(
    image=create_image(),
    gpu="A10G",
    timeout=600,  # 10 minutes timeout
    secrets=[modal.Secret.from_name("huggingface-secret")],
    network_file_systems={CACHE_DIR: cache}
)
class LLM:
    def __enter__(self):
        logger.info("Initializing LLM class...")
        self.hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            raise ValueError("HUGGINGFACE_TOKEN not found in environment")
        logger.info("Retrieved Hugging Face token from environment")
        
        logger.info("Logging in to Hugging Face...")
        login(token=self.hf_token)
        logger.info("Successfully logged in to Hugging Face")
        
        logger.info("Loading tokenizer from cache or downloading...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            cache_dir=CACHE_DIR
        )
        logger.info("Tokenizer loaded successfully")
        
        logger.info("Loading model from cache or downloading...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR
        )
        logger.info("Model loaded successfully")
        
        logger.info("Setting up pipeline...")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        logger.info("Pipeline setup complete")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    async def generate(self, prompt_text: str, history: List[Any]) -> str:
        logger.info("Starting text generation...")
        
        # Format the prompt with clear instructions
        logger.info("Formatting prompt with history...")
        system_prompt = """You are Bud, an experienced AI handyman assistant. Your role is to provide helpful, practical advice for home maintenance and repair issues. When users describe problems, provide clear, step-by-step solutions and safety precautions. Be thorough but conversational in your responses."""
        
        formatted_prompt = f"{system_prompt}\n\nUser: {prompt_text}\n\nBud:"
        logger.info("Prompt formatted successfully")
        
        logger.info("Generating response...")
        result = self.pipe(formatted_prompt, return_full_text=False)[0]['generated_text']
        logger.info("Response generated successfully")
        
        # Process and clean up the response
        response = result.strip()
        logger.info("Response processed and ready to return")
        return response

@stub.function(
    image=create_image(),
    gpu="A10G",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    network_file_systems={CACHE_DIR: cache},
    is_generator=False
)
async def chat(data: dict) -> str:
    logger.info("Chat function called")
    try:
        # Handle both form-data and JSON input
        logger.info(f"Received data: {data}")
        
        # Extract prompt from either form-data content or JSON prompt_text
        prompt_text = data.get("content") or data.get("prompt_text", "")
        raw_history = data.get("history", [])
        
        # Log input data for debugging
        logger.info(f"Extracted prompt_text: {prompt_text}")
        logger.info(f"Received history length: {len(raw_history)}")
        
        # Convert history format
        history = []
        for msg in raw_history:
            if isinstance(msg, str):
                # Handle string messages (from form-data)
                history.append(HumanMessage(content=msg))
            else:
                # Handle dict messages (from JSON)
                content = msg.get("content", "")
                msg_type = msg.get("type", "human")
                if msg_type == "human":
                    history.append(HumanMessage(content=content))
                else:
                    history.append(AIMessage(content=content))
        
        logger.info("Creating LLM instance...")
        with LLM() as llm:
            logger.info("LLM instance created")
            logger.info(f"Generating response for prompt: {prompt_text[:50]}...")
            response = await llm.generate(prompt_text, history)
            logger.info(f"Response generated successfully. Type: {type(response)}, Length: {len(str(response))}")
            logger.info(f"Response preview: {str(response)[:100]}...")
            return str(response)  # Ensure we return a string
    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}")
        logger.error(f"Full error details: {repr(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    stub.serve() 