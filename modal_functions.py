import modal
from modal import App, Image, Secret
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging
import os
from huggingface_hub import login

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_image():
    return (
        modal.Image.debian_slim()
        .pip_install([
            "torch",
            "transformers",
            "langchain",
            "langchain_core",
            "accelerate",
            "huggingface-hub",
            "einops",  # Required for TinyLlama
            "safetensors"  # Better model loading
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

app = modal.App("just-call-bud-prod")

@app.cls(
    image=create_image(),
    gpu="A10G",
    timeout=60,
    secrets=[modal.Secret.from_name("just_call_bud_secrets")]
)
class LLM:
    def __init__(self):
        logger.info("Initializing LLM class...")
        try:
            # Get Hugging Face token from Modal secrets
            hf_token = modal.Secret.from_name("just_call_bud_secrets")["HUGGINGFACE_TOKEN"]
            
            # Log in to Hugging Face
            logger.info("Logging in to Hugging Face...")
            login(token=hf_token)
            logger.info("Successfully logged in to Hugging Face")
            
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf",
                torch_dtype=torch.float16,
                token=hf_token
            )
            logger.info("Tokenizer loaded successfully")

            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf",
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token
            )
            logger.info("Model loaded successfully")

            logger.info("Setting up pipeline...")
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=500,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
            logger.info("Pipeline setup complete")
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    async def generate(self, prompt_text: str, history=None) -> str:
        try:
            logger.info("Starting text generation...")
            if history is None:
                history = []
            
            logger.info("Formatting prompt with history...")
            chain_input = {
                "history": history,
                "input": prompt_text
            }
            formatted_prompt = prompt.format_messages(**chain_input)
            
            full_prompt = "\n".join([msg.content for msg in formatted_prompt])
            logger.info("Prompt formatted successfully")
            
            logger.info("Generating response...")
            response = self.pipeline(full_prompt)[0]['generated_text']
            logger.info("Response generated successfully")
            
            new_content = response[len(full_prompt):].strip()
            logger.info("Response processed and ready to return")
            return new_content
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise

@app.function(
    image=create_image(),
    secrets=[modal.Secret.from_name("just_call_bud_secrets")]
)
async def chat(prompt_text: str, history=None) -> str:
    logger.info("Chat function called")
    try:
        llm = LLM()
        logger.info("LLM instance created")
        response = await llm.generate(prompt_text, history)
        logger.info("Response generated successfully")
        return response
    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}")
        raise

if __name__ == "__main__":
    print("=== Testing Modal Deployment ===")
    with app.run():
        try:
            response = chat.remote("How do I fix a leaky faucet?")
            print(f"Success! Response: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")
            raise 