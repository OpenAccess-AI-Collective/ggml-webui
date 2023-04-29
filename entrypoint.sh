#!/bin/bash

# fix the iframe viewport
echo "window.addEventListener('load',function(){var e=document.createElement('script');e.src='https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.2/iframeResizer.contentWindow.min.js',e.type='text/javascript',document.head.appendChild(e);});" >> main.js
sed -i 's/\(height: calc(100vh - 300px);\)/\/\*\1\*\//' css/html_cai_style.css
sed -i 's/\(height: calc(100vh - 300px);\)/\/\*\1\*\//' css/html_instruct_style.css
sed -i 's/\(height: calc(100vh - 300px);\)/\/\*\1\*\//' css/html_bubble_chat_style.css

# Activate the virtual environment and start the server
source /app/venv/bin/activate
python3 server.py --chat --listen --listen-port 7860 --model stable-vicuna-13B.ggml.q4_3.bin "$@"
