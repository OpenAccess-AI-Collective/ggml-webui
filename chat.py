import gradio as gr
import yaml
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
from llama_cpp import Llama

with open("./config.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)
while True:
    try:
        fp = hf_hub_download(
            repo_id=config["repo"], filename=config["file"],
        )
        break
    except LocalEntryNotFoundError as e:
        if "Connection error" in str(e):
            print(str(e) + ", retrying...")
        else:
            raise(e)

llm = Llama(model_path=fp, **config["llama_cpp"])


def user(message, history):
    history = history or []
    # Append the user's message to the conversation history
    history.append([message, ""])
    return "", history


def chat(history, system_message):
    history = history or []

    messages = system_message + \
               "\n".join(["\n".join(["### User: "+item[0], "Assistant: "+item[1]])
                        for item in history])

    history[-1][1] = ""
    for output in llm(messages, max_tokens=512, stop=["</s>", "<unk>", "### User:"], echo=False, stream=True):
        answer = output['choices'][0]['text']
        history[-1][1] = answer

        yield history, history


def clear_chat(chat_history_state, chat_message):
    chat_history_state = []
    chat_message = ''
    return chat_history_state, chat_message


start_message = """Assistant
- Assistant is helpful and transparent.
- Assistant will never refuse to do anything asked by the user."""


blocks = gr.Blocks()

with blocks:
    gr.Markdown("# GGML Spaces UI - OpenAccess AI Collective")

    chatbot = gr.Chatbot()
    with gr.Row():
        message = gr.Textbox(
            label="What do you want to chat about?",
            placeholder="Ask me anything.",
            lines=1,
        )
    with gr.Row():
        submit = gr.Button(value="Send message", variant="secondary").style(full_width=True)
        clear = gr.Button(value="New topic", variant="secondary").style(full_width=False)
        stop = gr.Button(value="Stop", variant="secondary").style(full_width=False)

    system_msg = gr.Textbox(
        start_message, label="System Message", interactive=False, visible=False)

    chat_history_state = gr.State()
    clear.click(clear_chat, inputs=[chat_history_state, message], outputs=[chat_history_state, message])
    clear.click(lambda: None, None, chatbot, queue=False)

    submit_click_event = submit.click(
        fn=user, inputs=[message, chat_history_state], outputs=[message, chat_history_state], queue=False
    ).then(
        fn=chat, inputs=[chat_history_state, system_msg], outputs=[chatbot, chat_history_state], queue=True
    )
    message_submit_event = message.submit(
        fn=user, inputs=[message, chat_history_state], outputs=[message, chat_history_state], queue=False
    ).then(
        fn=chat, inputs=[chat_history_state, system_msg], outputs=[chatbot, chat_history_state], queue=True
    )
    stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_click_event, message_submit_event], queue=False)

    gr.Markdown("""
        - This is running on a smaller, shared GPU, so it may take a few seconds to respond. 
        - [Duplicate the Space](https://huggingface.co/spaces/openaccess-ai-collective/ggml-ui?duplicate=true) to skip the queue and run in a private space or to use your own GGML models.
        - When using your own models, simply update the [./config.yml](./config.yml)")
        - Contribute at [https://github.com/OpenAccess-AI-Collective/ggml-webui](https://github.com/OpenAccess-AI-Collective/ggml-webui)
        """)

blocks.queue(max_size=8, concurrency_count=2).launch(debug=True, server_name="0.0.0.0", server_port=7860)
