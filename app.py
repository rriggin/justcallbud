from flask import Flask, render_template, request, jsonify, url_for
from datetime import datetime
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Store messages in memory (replace with database in production)
messages = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/messages', methods=['GET'])
def get_messages():
    return jsonify(messages if messages else [])

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            print("Invalid request data:", data)  # Debug log
            return jsonify({'error': 'Invalid request'}), 400
            
        user_message = data['content']
        print(f"Received message: {user_message}")  # Debug log
        
        # Store user message
        user_msg = {
            'content': user_message,
            'isUser': True,
            'timestamp': datetime.now().isoformat()
        }
        messages.append(user_msg)
        
        print("Calling OpenAI API...")  # Debug log
        # Generate AI response using OpenAI
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Bud, a friendly and knowledgeable AI assistant specializing in home maintenance and repairs. Provide helpful, practical advice while maintaining a conversational tone."},
                {"role": "user", "content": user_message}
            ]
        )
        
        ai_response = completion.choices[0].message.content
        print(f"Got AI response: {ai_response[:100]}...")  # Debug log
        
        # Store AI response
        ai_msg = {
            'content': ai_response,
            'isUser': False,
            'timestamp': datetime.now().isoformat()
        }
        messages.append(ai_msg)
        
        return jsonify(ai_msg)
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")  # More detailed error logging
        import traceback
        print(traceback.format_exc())  # Print full stack trace
        error_msg = {
            'content': "I apologize, but I'm having trouble processing your request right now. Please try again later.",
            'isUser': False,
            'timestamp': datetime.now().isoformat()
        }
        messages.append(error_msg)
        return jsonify(error_msg)

@app.route('/api/messages', methods=['DELETE'])
def clear_messages():
    global messages
    messages = []
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY not found in environment variables")
    app.run(debug=True)