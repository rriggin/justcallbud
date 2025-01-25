import modal
import sys

def check_modal():
    print("=== Modal Status ===")
    print(f"Token present: {bool(modal.token.get_token())}")
    
    try:
        app = modal.App.lookup("just-call-bud-prod")
        print("\n=== Available Functions ===")
        print(dir(app))
        
        # Try to call test function
        print("\n=== Testing Connection ===")
        with app.run():
            result = app.test.remote()
            print(f"Test result: {result}")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    check_modal() 