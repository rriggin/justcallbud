import warnings
import urllib3
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)

import modal
import time
import subprocess
import os
from modal import Stub, Image, Secret

def create_image():
    return (
        modal.Image.debian_slim()
        .apt_install("curl")
        .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
        .pip_install(["requests"])
    )

app = modal.App("just-call-bud-prod", image=create_image())

@app.function(
    gpu="T4",
    secrets=[modal.Secret.from_name("just-call-bud-secrets")]
)
async def get_llama_response(prompt: str):
    import requests
    import subprocess
    import time
    
    # Start Ollama and wait for it
    subprocess.Popen(['ollama', 'serve'])
    time.sleep(5)  # Wait for Ollama to start
    
    try:
        # Pull model if not exists
        subprocess.run(['ollama', 'pull', 'llama2'])
        
        # Make the request
        response = requests.post('http://localhost:11434/api/generate', 
            json={
                "model": "llama2",
                "prompt": prompt,
                "stream": False
            }, timeout=60)
        
        result = response.json()
        print(f"Llama response: {result}")
        return result['response']
        
    except Exception as e:
        print(f"Error in get_llama_response: {str(e)}")
        raise

@app.function()
async def test():
    return "Modal connection successful!"

# For testing
if __name__ == "__main__":
    with app.run():
        response = get_llama_response.remote("Hi, how are you?")
        print(f"Test response: {response}")

# The test function can still be used by other parts of the app 