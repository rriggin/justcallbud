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

@app.function()  # Test function without GPU
async def test():
    return "Modal connection successful!"

@app.function(gpu="T4")  # Main function with GPU
async def get_llama_response(prompt: str):
    return f"Test response to: {prompt}"

# For testing
if __name__ == "__main__":
    with app.run():
        print("Testing connection...")
        test_result = test.remote()
        print(f"Test result: {test_result}")

# The test function can still be used by other parts of the app 