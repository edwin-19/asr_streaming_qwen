import gradio as gr
import numpy as np
import websockets
import json
import asyncio

WS_URL = "ws://localhost:8000/ws/transcribe"

async def stream_transcribe(stream_state, new_chunk):
    if new_chunk is None:
        return stream_state, stream_state.get("transcript", "")

    sr, y = new_chunk
    
    # 1. Processing (Mono + Float32 + Normalization)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    
    if np.max(np.abs(y)) > 0:
        y /= np.max(np.abs(y))

    # 2. Connection Management (Websockets 15.x Style)
    ws = stream_state.get("ws")
    
    # In 15.x, we check if the connection exists and isn't closing/closed
    is_connected = False
    if ws is not None:
        # Check if the connection state is OPEN
        # The .state property is the most reliable way in 15.x
        is_connected = hasattr(ws, 'state') and str(ws.state).split('.')[-1] == "OPEN"

    try:
        if not is_connected:
            stream_state["ws"] = await websockets.connect(WS_URL)
            stream_state["transcript"] = ""
            ws = stream_state["ws"]

        # 3. Send & Receive
        await ws.send(y.tobytes())
        response = await ws.recv()
        data = json.loads(response)
        
        # 4. Update and Return
        stream_state["transcript"] = data.get("text", "")
        return stream_state, stream_state["transcript"]

    except Exception as e:
        # If any socket error occurs, null it out so we reconnect on next chunk
        stream_state["ws"] = None
        return stream_state, f"Reconnecting... ({str(e)})"

# --- UI Setup ---
with gr.Blocks() as demo:
    state = gr.State(value={"ws": None, "transcript": ""})
    
    gr.Markdown("# 🚀 High-Speed ASR Stream")
    gr.Markdown(f"Connected to: `{WS_URL}`")
    
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], streaming=True)
        text_output = gr.Textbox(label="Live Transcript", interactive=False)

    # Use a faster stream interval if your server can handle it (0.5s is default)
    audio_input.stream(
        fn=stream_transcribe,
        inputs=[state, audio_input],
        outputs=[state, text_output],
        show_progress="hidden"
    )

if __name__ == "__main__":
    demo.launch(share=True)