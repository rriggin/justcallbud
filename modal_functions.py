import modal
from modal import App, Image, Secret
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def create_image():
    return (
        modal.Image.debian_slim()
        .pip_install([
            "torch",
            "transformers",
            "langchain",
            "langchain_core",
            "accelerate"
        ])
    )

SYSTEM_PROMPT = """You are Bud, a friendly and knowledgeable AI handyman assistant. 
Core rules:
1. Never use asterisks (*) or describe actions
2. Never use emotes or roleplay elements
3. Maintain a professional, direct tone
4. Focus only on home repair advice
5. Start responses with "Hello" or direct answers
6. Avoid casual expressions or emojis

Your purpose is to provide practical, safety-focused home maintenance advice.
Keep all responses focused on technical details and solutions."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

app = modal.App("just-call-bud-prod")

@app.cls(
    image=create_image(),
    gpu="A10G",
    timeout=60,
    secrets=[modal.Secret.from_name("just_call_bud_secrets")]
)
class LLM:
    def __enter__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )

    async def generate(self, prompt_text: str, history=None) -> str:
        if history is None:
            history = []
            
        # Format the prompt with history using LangChain
        chain_input = {
            "history": history,
            "input": prompt_text
        }
        formatted_prompt = prompt.format_messages(**chain_input)
        
        # Convert formatted messages to a single string
        full_prompt = "\n".join([msg.content for msg in formatted_prompt])
        
        # Generate response
        response = self.pipeline(full_prompt)[0]['generated_text']
        
        # Extract only the new content after the prompt
        new_content = response[len(full_prompt):].strip()
        return new_content

@app.function(
    image=create_image(),
    secrets=[modal.Secret.from_name("just_call_bud_secrets")]
)
async def chat(prompt_text: str, history=None) -> str:
    llm = LLM()
    response = await llm.generate(prompt_text, history)
    return response

if __name__ == "__main__":
    print("=== Testing Modal Deployment ===")
    with app.run():
        try:
            response = chat.remote("How do I fix a leaky faucet?")
            print(f"Success! Response: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")
            raise 