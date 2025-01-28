# Just Call Bud - AI Home Maintenance Assistant

An AI-powered handyman assistant that helps users diagnose and fix home maintenance issues using local LLM technology.

## Project Overview
Just Call Bud is a web application that connects users with an AI assistant specialized in home maintenance and repairs. Users can describe their issues, upload photos, and receive practical advice and solutions.

## Current Features
- 💬 Real-time chat interface
- 📸 Image upload capability
- 🤖 Local LLM integration via Ollama
- �  Modal back end for Llama usage
- 📱 Responsive design
- ⌨️ Live typing indicators
- 🔄 Message history management

## Todos
-  build functionality for the ad space
-  add lead gen buttons in line, connect to a live handyman?
-  make facebook page and integrate llama chat message thing
-  optimize deployment:
   - speed up Render deployment:
     - investigate slow git clone/checkout
     - optimize dependency installation
     - analyze which ML packages can be safely removed
     - implement better dependency management
     - consider Docker pre-build
   - investigate Render deploy hook reliability
   - optimize model loading:
     - implement model caching
     - reduce cold start time

### Tech Stack
- **AI Model**: Meta's Llama 2 (7B parameters)
- **Compute**: Modal with GPU support
- **Frontend**: Flask + HTML/CSS
- **Deployment**: GitHub Actions → Modal → Render

### Development
Requires:
- Modal account
- Hugging Face account with Llama 2 access
- Render account
