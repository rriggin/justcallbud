import modal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_modal():
    try:
        logger.info("=== Checking Modal Setup ===")
        
        # Look up the app
        app = modal.App.lookup("just-call-bud-prod")
        logger.info(f"Found app: {app}")
        
        # Check functions
        logger.info(f"Registered functions: {app.registered_functions}")
        
        # Try to get the function
        if hasattr(app, 'get_llama_response'):
            logger.info("Found get_llama_response function")
            
            # Try to call it
            with app.run():
                response = app.get_llama_response.remote("test message")
                logger.info(f"Test response: {response}")
        else:
            logger.error("get_llama_response function not found")
            
    except Exception as e:
        logger.error(f"Error checking Modal: {str(e)}")

if __name__ == "__main__":
    check_modal() 