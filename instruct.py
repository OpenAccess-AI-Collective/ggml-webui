import gradio as gr
import yaml
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

with open("./config.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)
fp = hf_hub_download(
    repo_id=config["repo"], filename=config["file"],
)

llm = Llama(model_path=fp)

def generate_text(input_text):
    output = llm(f"### Instruction: {input_text}\n\n### Response: ",  echo=False, **config['chat'])
    return output['choices'][0]['text']

input_text = gr.inputs.Textbox(lines= 10, label="Enter your input text")
output_text = gr.outputs.Textbox(label="Output text")

description = f"""
### brought to you by OpenAccess AI Collective
- This is the [{config["repo"]}](https://huggingface.co/{config["repo"]}) model file [{config["file"]}](https://huggingface.co/{config["repo"]}/blob/main/{config["file"]})
- This Space uses GGML with GPU support, so it can quickly run larger models on smaller GPUs & VRAM.
- This is running on a smaller, shared GPU, so it may take a few seconds to respond.
- Due to a [missing feature in Gradio](https://github.com/gradio-app/gradio/issues/3914), the chatbot interface will not show you your status in the queue. If it's stuck, be patient.  
- [Duplicate the Space](https://huggingface.co/spaces/openaccess-ai-collective/ggml-ui?duplicate=true) to skip the queue and run in a private space or to use your own GGML models.
- When using your own models, simply update the [config.yml](https://huggingface.co/spaces/openaccess-ai-collective/ggml-ui/blob/main/config.yml)
- You can use instruct or chatbot mode by updating the README.md to either `app_file: instruct.py` or `app_file: chat.py`
- Contribute at [https://github.com/OpenAccess-AI-Collective/ggml-webui](https://github.com/OpenAccess-AI-Collective/ggml-webui)
"""

gr.Interface(
    fn=generate_text,
    inputs=input_text,
    outputs=output_text,
    title="GGML UI Demo",
    description=description,
).queue(max_size=16, concurrency_count=1).launch()
