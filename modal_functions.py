import warnings
import urllib3
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)

import modal
import time
import subprocess
import os
from modal import App, Image, Secret
import requests

def create_image():
    return modal.Image.debian_slim()

app = modal.App("just-call-bud-prod")

stub = modal.Stub()
MODEL = modal.Image.from_registry("ghcr.io/modal-labs/llama2-7b-chat-hf-q4f32_1")

@stub.cls(gpu="A10G", image=MODEL)
class LLM:
    def __enter__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalGeneration
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        self.model = AutoModelForCausalGeneration.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        
    def generate(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_length=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.function(
    image=MODEL,
    gpu="A10G",
    timeout=120
)
async def get_llama_response(prompt: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalGeneration
    
    formatted_prompt = f"""You are Bud, a friendly and knowledgeable AI handyman assistant. 
    You help people with home maintenance and repair questions.
    You provide clear, practical advice and always prioritize safety.
    
    User: {prompt}
    
    Assistant: """
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalGeneration.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    
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