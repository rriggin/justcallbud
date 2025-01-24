import warnings
import urllib3
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)

import modal
import time
import subprocess
import os

# Define the image
def create_image():
    return (
        modal.Image.debian_slim()
        .apt_install("curl")
        .run_commands(
            "curl -fsSL https://ollama.com/install.sh | sh",
        )
        .pip_install(["requests"])
    )

# Create the app with the image
app = modal.App("just-call-bud-prod", image=create_image())

@app.function(
    gpu="T4",
    secrets=[modal.Secret.from_name("just-call-bud-secrets")]
)
def get_llama_response(prompt: str):
    import requests
    
    # Access the secrets we actually have
    flask_env = os.environ["FLASK_ENV"]
    port = os.environ["PORT"]
    
    # Start Ollama and wait for it
    subprocess.Popen(['ollama', 'serve'])
    time.sleep(5)  # Wait for Ollama to start
    
    # Pull model if not exists
    subprocess.run(['ollama', 'pull', 'llama2'])
    
    # Now make the request
    response = requests.post('http://localhost:11434/api/generate', 
        json={
            "model": "llama2",
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 2048,
                "num_thread": 8,
                "temperature": 0.7,
                "top_k": 20,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }, timeout=60)
    
    return response.json()['response']

@app.local_entrypoint()
def main():
    # No need for app.run() context when using local_entrypoint
    result = get_llama_response.remote("Hi, how are you?")
    print(f"Test response: {result}")

# The test function can still be used by other parts of the app
@app.function()
def test():
    return get_llama_response.remote("Hi, how are you?") 