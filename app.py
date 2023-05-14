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
    output = llm(f"### Instruction: {input_text}\n\n### Response: ", max_tokens=256, stop=["</s>", "<unk>", "### Instruction:"], echo=True)
    return output['choices'][0]['text']

input_text = gr.inputs.Textbox(lines= 10, label="Enter your input text")
output_text = gr.outputs.Textbox(label="Output text")

description = f"""llama.cpp implementation in python [https://github.com/abetlen/llama-cpp-python]

This is the {config["repo"]}/{config["file"]} model.
"""

examples = [
    ["Tell me a joke about old houses.", "Why did the old house break up with the new house? Because it was too modern!"],
    ["What is the square root of 64?", "The square root of 64 is 8."],
    ["Insult me", ""],
]

gr.Interface(fn=generate_text, inputs=input_text, outputs=output_text, title="Llama Language Model", description=description, examples=examples).launch()

