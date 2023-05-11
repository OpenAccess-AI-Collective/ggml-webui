#!/bin/bash

# fix the iframe viewport
echo "window.addEventListener('load',function(){var e=document.createElement('script');e.src='https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.2/iframeResizer.contentWindow.min.js',e.type='text/javascript',document.head.appendChild(e);});" >> main.js
sed -i 's/\(height: calc(100vh - 306px);\)/\/\*\1\*\//' css/chat_style-cai-chat.css
sed -i 's/\(height: calc(100vh - 306px);\)/\/\*\1\*\//' css/html_instruct_style.css
sed -i 's/\(height: calc(100vh - 306px);\)/\/\*\1\*\//' css/chat_style-wpp.css

# Activate the virtual environment and start the server
source /app/venv/bin/activate
export MPLCONFIGDIR=/tmp/matplotlib
model_name=$(python -c "import yaml;print(yaml.safe_load(open('models.yml'))['model_name'])")
model_base=$(python -c "import yaml;print(yaml.safe_load(open('models.yml'))['model_base'])")
mkdir -p /app/models/$model_name
python3 download-model.py $model_base --text-only --output /app/models/
url=$(python -c "import yaml;print(yaml.safe_load(open('models.yml'))['url'])")
curl -L $url -o "/app/models/$model_name/$(basename $url)"
rm /app/models/$model_name/*.index.json
ls -l /app/models/$model_name/
ls -l /app
num_cpus=$(nproc)
python3 server_spaces.py --chat --listen --listen-port 7860 --auto-devices --model "$model_name/$(basename $url)" --threads $num_cpus
