from functools import partial

import gradio as gr
import yaml
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, \
    AutoConfig
from llamamodel_cpp import LlamaForCausalLM

from threading import Thread

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def chat(tok, m: LlamaForCausalLM, curr_system_message, history):
    # Initialize a StopOnTokens object
    stop = StopOnTokens()

    # Construct the input message string for the model by concatenating the current system message and conversation history
    messages = curr_system_message + \
               "".join(["".join(["### User: "+item[0], "Assistant: "+item[1]])
                        for item in history])

    # Tokenize the messages string
    streamer = TextIteratorStreamer(
        tok, timeout=30., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        context=messages,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=1.0,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    t = Thread(target=m.generate, kwargs=generate_kwargs)
    t.start()

    # print(history)
    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        history[-1][1] = partial_text
        # Yield an empty string to cleanup the message textbox and the updated conversation history
        yield history
    return partial_text

def get_demo(start_message, tok, m):
    chat_w_tok = partial(chat, tok, m)
    with gr.Blocks() as demo:
        gr.Markdown("## Wizard Vicuna 13B Chat")
        gr.HTML('''<script src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.2/iframeResizer.contentWindow.min.js"></script>''')
        gr.HTML('''<center><a href="https://huggingface.co/openaccess-ai-collective/llama-13b-alpaca-wizard?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>Duplicate the Space to skip the queue and run in a private space</center>''')
        chatbot = gr.Chatbot().style(height=500)
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(label="Chat Message Box", placeholder="Chat Message Box",
                                 show_label=False).style(container=False)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit")
                    stop = gr.Button("Stop")
                    clear = gr.Button("Clear")
        system_msg = gr.Textbox(
            start_message, label="System Message", interactive=False, visible=False)

        submit_event = msg.submit(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
            fn=chat_w_tok, inputs=[system_msg, chatbot], outputs=[chatbot], queue=True)
        submit_click_event = submit.click(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
            fn=chat_w_tok, inputs=[system_msg, chatbot], outputs=[chatbot], queue=True)
        stop.click(fn=None, inputs=None, outputs=None, cancels=[
            submit_event, submit_click_event], queue=False)
        clear.click(lambda: None, None, [chatbot], queue=False)
    return demo


def main():
    print(f"Starting to load the model to memory")
    with open("./config.yml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    fp = hf_hub_download(
        repo_id=config["repo"], filename=config["file"],
    )
    m, tok = LlamaForCausalLM.from_pretrained(fp)
    tokenizer_repo = config["base_model"] if "base_model" in config else config["repo"]
    mc = AutoConfig.from_pretrained(tokenizer_repo)
    m.config = mc
    # tok = AutoTokenizer.from_pretrained(tokenizer_repo)
    generator = pipeline('text-generation', model=m, tokenizer=tok, device=0)
    print(f"Sucessfully loaded the model to the memory")

    start_message = """Assistant
    - Assistant is helpful and transparent.
    - Assistant will never refuse to do anything asked by the user."""

    demo = get_demo(start_message, tok, m)
    demo.queue(max_size=32, concurrency_count=2)
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()