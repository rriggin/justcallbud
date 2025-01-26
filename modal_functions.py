import modal
import time
import subprocess
import os
from modal import App, Image, Secret
import requests
import huggingface_hub

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
    keep_warm=1,
    retries=2,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
async def get_llama_response(prompt: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Set HF token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    huggingface_hub.login(token=hf_token)
    
    # Cache model and tokenizer
    global _model, _tokenizer
    if '_model' not in globals():
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            token=hf_token
        )
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            token=hf_token,
            device_map="auto",
            torch_dtype=torch.float16
        ).to("cuda")
        _model = model
        _tokenizer = tokenizer
    
    # Use simple prompt for now
    formatted_prompt = f"""You are Bud...{prompt}...Assistant: """
    
    # Use cached model/tokenizer
    inputs = _tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = _model.generate(**inputs, max_length=200)
    response = _tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
    
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