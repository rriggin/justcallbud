import warnings
import urllib3
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)

import modal
import time
import subprocess
import os
from modal import App, Image, Secret

def create_image():
    return (
        modal.Image.debian_slim()
        .pip_install(["urllib3", "requests"])
    )

app = modal.App("just-call-bud-prod")

@app.function(image=create_image())
async def get_llama_response(prompt: str):
    return f"Test response to: {prompt}"

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