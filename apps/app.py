import gradio as gr
from model_inference import generate_stream_for_chatbox
from llmbrew.utils import Logger
logger=Logger.get_logger()

def generate_text(message,history,temperature,top_k,max_new_tokens):
    logger.info(f"message:{message}")
    logger.info(f"history:{history}")
    logger.info(f"temperature:{temperature},top_k:{top_k},max_new_tokens:{max_new_tokens}")
    generator =generate_stream_for_chatbox(prompt=message,
                                    do_sample= True,
                                  temperature= float(temperature),
                                  top_k= int(top_k),
                                  max_new_tokens = int(max_new_tokens))
    text=""
    for token in generator:
        text+=token
        yield text

with gr.Blocks(title="llmbrew chatbox") as app:
    with gr.Row():
        gr.Markdown("<h2 align='center'>LLMBrew Chatbox</h2>")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## parameter settings")
            temperature=gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="temperature")
            top_k=gr.Slider(1, 100, value=40, step=1, label="top_k")
            max_new_tokens=gr.Slider(10, 300, value=100, step=10, label="max_new_tokens")
        with gr.Column(scale=2):
            gr.Markdown('''
              **llmbrew** · 10.09M params · 4 layers · hidden 320 · 5 heads · 512 context · RoPE + RMSNorm + SwiGLU
            ''')
            gr.ChatInterface(
                    fn=generate_text,
                    chatbot=gr.Chatbot(
                        height=400,
                        placeholder="I am LLMBrew，what can I help you？",
                    ),
                    textbox=gr.Textbox(
                        placeholder="please send message……",
                        container=False,
                    ),
                    title="LLMBrew Chat",
                    additional_inputs=[
                        temperature,
                        top_k,
                        max_new_tokens
                        ]
                )
app.queue(max_size=32,default_concurrency_limit=1)
app.launch(
    server_name="127.0.0.1",
    server_port=7860,
    inbrowser=True,
    share=False,
    show_error=True,
    auth=None,
    prevent_thread_lock=False,
)
