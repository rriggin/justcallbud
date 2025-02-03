import modal
from modal import App, Image, Secret
import datetime

def create_image():
    return modal.Image.debian_slim().pip_install("requests")

app = modal.App("just-call-bud-prod")

@app.function(
    image=create_image(),
    timeout=60,
    secrets=[modal.Secret.from_name("just_call_bud_secrets")]
)
async def test_deployment():
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"Deployment test successful at {timestamp}"

if __name__ == "__main__":
    print("=== Testing Modal Deployment ===")
    with app.run():
        try:
            response = test_deployment.remote()
            print(f"Success! Response: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")
            raise 