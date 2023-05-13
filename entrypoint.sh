#!/bin/bash

source /app/venv/bin/activate
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python3 app.py
