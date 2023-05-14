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


def chat(inp, history, system_message):
    history = history or []
    history.append((inp, ""))

    messages = system_message + \
               "\n".join(["\n".join(["### User: "+item[0], "Assistant: "+item[1]])
                        for item in history])

    history = history or []

    output = llm(messages, max_tokens=512, stop=["</s>", "<unk>", "### User:"], echo=False)
    answer = output['choices'][0]['text']

    history.pop()  # remove user input only history
    history.append((inp, answer))

    message = '' # This clears the message text

    return history, history, message


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
        clear = gr.Button(value="New topic", variant="secondary").style(full_width=False)
    submit = gr.Button(value="Send message", variant="secondary").style(full_width=True)

    system_msg = gr.Textbox(
        start_message, label="System Message", interactive=False, visible=False)

    # gr.Examples(
    #     examples=[
    #         "Tell me a joke about old houses.",
    #         "Insult me.",
    #         "What is the future of AI and large language models?",
    #     ],
    #     inputs=message,
    # )

    chat_history_state = gr.State()
    clear.click(clear_chat, inputs=[chat_history_state, message], outputs=[chat_history_state, message])
    clear.click(lambda: None, None, chatbot, queue=False)

    submit.click(chat, inputs=[message, chat_history_state, system_msg], outputs=[chatbot, chat_history_state, message])
    message.submit(chat, inputs=[message, chat_history_state, system_msg], outputs=[chatbot, chat_history_state, message])

blocks.queue(concurrency_count=10).launch(debug=True)