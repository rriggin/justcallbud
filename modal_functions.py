import warnings
import urllib3
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)

import modal
import time
import subprocess
import os
from modal import Stub, Image, Secret

def create_image():
    return modal.Image.debian_slim().pip_install(["requests"])

app = modal.App("just-call-bud-prod")

@app.function(gpu="T4")
async def get_llama_response(prompt: str):
    return f"Test response to: {prompt}"  # Simple test response

@app.function()
async def test():
    return "Modal connection successful!"

# For testing
if __name__ == "__main__":
    with app.run():
        response = get_llama_response.remote("Hi, how are you?")
        print(f"Test response: {response}")

# The test function can still be used by other parts of the app 