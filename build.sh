#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Installing Python packages..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Verify transformers installation
echo "Verifying transformers installation..."
python -c "import transformers; print(f'transformers version: {transformers.__version__}')"

echo "Build completed successfully!" 