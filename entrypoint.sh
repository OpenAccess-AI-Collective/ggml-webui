#!/bin/bash

# Activate the virtual environment and start the server
source /app/venv/bin/activate
python3 server.py --chat --listen --listen-port 7860 --model stable-vicuna-13B.ggml.q4_3.bin "$@"
