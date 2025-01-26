import modal
import time
import subprocess
import os
from modal import App, Image, Secret
import requests

def create_image():
    return (
        modal.Image.debian_slim()
        .pip_install([
            "torch",
            "transformers",
            "accelerate",
            "safetensors",
            "requests",
            "urllib3",
            "huggingface-hub"
        ])
    )

app = modal.App("just-call-bud-prod")

@app.function(
    image=create_image(),
    gpu="A10G",
    timeout=120,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
async def get_llama_response(prompt: str):
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM
    )
    
    # Set HF token before loading model
    os.environ["HUGGINGFACE_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN")
    
    formatted_prompt = f"""You are Bud, a friendly and knowledgeable AI handyman assistant. 
    You help people with home maintenance and repair questions.
    You provide clear, practical advice and always prioritize safety.
    
    User: {prompt}
    
    Assistant: """
    
    # Now load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    
    # Generate response
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

@app.function(image=create_image())
async def test_function():
    """Test if functions are being registered"""
    return "Test function works"

if __name__ == "__main__":
    print("=== Testing Modal Deployment ===")
    with app.run():
        try:
            response = get_llama_response.remote("test")
            print(f"Success! Response: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")
            raise 