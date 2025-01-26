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
    formatted_prompt = f"""You are Bud, a friendly and knowledgeable AI handyman assistant. 
    As a handyman with decades of experience, you:
    - Speak in a warm, conversational tone
    - Share personal insights from your experience
    - Vary your greetings and responses naturally
    - Focus on practical, safety-first solutions
    - Avoid repeating the same phrases
    
    User: {prompt}
    
    Assistant: """
    
    # Use cached model/tokenizer
    inputs = _tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = _model.generate(
        **inputs,
        max_length=400,         # Increase max length
        min_length=50,          # Ensure substantive responses
        temperature=0.7,        # Increase creativity
        top_p=0.9,             # Nucleus sampling
        repetition_penalty=1.2, # Discourage repetition
        no_repeat_ngram_size=3, # Avoid repeating phrases
        early_stopping=True     # Stop when complete
    )
    response = _tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
    
    # Extract only the assistant's response
    # Split at both Assistant: and User: to avoid hallucinated user responses
    parts = response.split("Assistant: ")[-1].split("User:")[0].strip()
    
    return parts

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