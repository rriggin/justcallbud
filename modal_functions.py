import warnings
import urllib3
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)

import modal
import time
import subprocess
import os
from modal import App, Image, Secret

def create_image():
    return modal.Image.debian_slim().pip_install(["requests"])

app = modal.App("just-call-bud-prod")

@app.function()
async def get_llama_response(prompt: str):
    return f"Test response to: {prompt}"

# For testing
if __name__ == "__main__":
    print("=== Testing Modal Deployment ===")
    print(f"App name: {app.name}")
    print(f"Function registered: {'get_llama_response' in app.registered_functions}")
    
    with app.run():
        try:
            response = get_llama_response.remote("test")
            print(f"Success! Response: {response}")
            
            # Check if function is registered after run
            print("\n=== Post-run check ===")
            deployed_app = modal.App.lookup("just-call-bud-prod")
            print(f"Found app: {deployed_app}")
            print(f"Functions: {deployed_app.registered_functions}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            raise 