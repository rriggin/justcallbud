# Start with Ubuntu base image instead of ollama directly
FROM ubuntu:22.04

# Install Python, pip, and other dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Start Ollama and pull the model
RUN ollama serve & sleep 5 && ollama pull llama2 