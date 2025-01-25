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
    return (
        modal.Image.debian_slim()
        .pip_install(["requests"])
    )

app = modal.App("just-call-bud-prod")

@app.function(image=create_image())
async def get_llama_response(prompt: str):
    # Format the prompt for our handyman assistant
    formatted_prompt = f"""You are Bud, a friendly and knowledgeable AI handyman assistant. 
    You help people with home maintenance and repair questions.
    You provide clear, practical advice and always prioritize safety.
    
    User: {prompt}
    
    Assistant: """
    
    # Call Ollama API
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            "model": "llama2",
            "prompt": formatted_prompt,
            "stream": False
        }
    ).json()
    
    return response['response']

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