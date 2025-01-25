import modal
import sys
import os
from modal.cli import token

def check_modal():
    print("=== Modal Status ===")
    try:
        token_obj = token.get_token()
        print(f"Token present: {bool(token_obj)}")
        if token_obj:
            print(f"Token ID: {token_obj.id}")
    except Exception as e:
        print(f"Token error: {str(e)}")
    
    try:
        print("\n=== Looking up app ===")
        app = modal.App.lookup("just-call-bud-prod")
        print(f"App found: {app}")
        
        print("\n=== App Details ===")
        print(f"App name: {app.name}")
        print(f"App functions: {app.registered_functions}")
        print(f"App dir: {dir(app)}")
        
        # Try to get function directly
        print("\n=== Function Access ===")
        if hasattr(app, 'get_llama_response'):
            print("Found get_llama_response via hasattr")
        else:
            print("Could not find get_llama_response via hasattr")
            
        try:
            func = app.registered_functions['get_llama_response']
            print(f"Found function: {func}")
        except KeyError:
            print("Function not in registered_functions")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    check_modal() 