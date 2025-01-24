import warnings
import urllib3
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)

from modal import Image, Secret, Stub, web_endpoint

# Define the image
def create_image():
    return (
        Image.debian_slim()
        .apt_install("curl")
        .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
        .pip_install(["requests"])
    )

# Create the stub with the image
stub = Stub("just-call-bud-prod", image=create_image())

@stub.function(
    gpu="T4",
    secrets=[Secret.from_name("just-call-bud-secrets")]
)
async def get_llama_response(prompt: str):
    import requests
    
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

# For testing
if __name__ == "__main__":
    with stub.run():
        response = get_llama_response.remote("Hi, how are you?")
        print(f"Test response: {response}")

# The test function can still be used by other parts of the app
@stub.function()
async def test():
    return "Modal connection successful!"  # Simple response to verify connection 