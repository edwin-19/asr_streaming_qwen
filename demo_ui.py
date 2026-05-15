import gradio as gr
import numpy as np
import websockets
import json
import asyncio

WS_URL = "ws://localhost:8000/ws/transcribe"
BUFFER_LIMIT_SEC = 5.0  # Keep 5 seconds of context for the model
SAMPLE_RATE = 16000

async def stream_transcribe(stream_state, new_chunk):
    if new_chunk is None:
        return stream_state, stream_state.get("transcript", "")

    sr, y = new_chunk
    
    # 1. Processing (Mono + Float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    
    # 2. Accumulate in local Buffer
    if "buffer" not in stream_state:
        stream_state["buffer"] = y
    else:
        # Append new audio to existing buffer
        stream_state["buffer"] = np.concatenate([stream_state["buffer"], y])
    
    # 3. Limit Buffer Size (Sliding Window)
    # This ensures we don't send 10 minutes of audio at once, 
    # but still give the model context.
    max_samples = int(BUFFER_LIMIT_SEC * sr)
    if len(stream_state["buffer"]) > max_samples:
        stream_state["buffer"] = stream_state["buffer"][-max_samples:]

    # 4. Connection Management (Websockets 15.x)
    ws = stream_state.get("ws")
    is_connected = ws is not None and hasattr(ws, 'state') and str(ws.state).split('.')[-1] == "OPEN"

    try:
        if not is_connected:
            stream_state["ws"] = await websockets.connect(WS_URL)
            ws = stream_state["ws"]

        # 5. Send the BUFFER, not just the chunk
        # Sending the last 5 seconds helps Whisper 'catch up' on context
        await ws.send(stream_state["buffer"].tobytes())
        
        response = await ws.recv()
        data = json.loads(response)
        
        # Only update if the server actually returned text
        if data.get("text"):
            stream_state["transcript"] = data["text"]
            
        return stream_state, stream_state["transcript"]

    except Exception as e:
        stream_state["ws"] = None
        return stream_state, f"Buffer Syncing... ({str(e)})"

with gr.Blocks() as demo:
    # State now tracks the socket, the full transcript, and the audio buffer
    state = gr.State(value={"ws": None, "transcript": "", "buffer": np.array([], dtype=np.float32)})
    
    audio_input = gr.Audio(sources=["microphone"], streaming=True)
    text_output = gr.Textbox(label="Live Transcription (5s Context Window)")

    audio_input.stream(
        fn=stream_transcribe,
        inputs=[state, audio_input],
        outputs=[state, text_output],
        show_progress="hidden"
    )

demo.launch()