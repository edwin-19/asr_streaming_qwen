import json
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import re

socket_router = APIRouter(prefix="/ws", tags=["socket-streaming-asr"])

def update_fixed_window_metadata_ws(state, context):
    full_text = getattr(state, "text", "") or ""
    current_time = getattr(state, "chunk_id", 0) * 0.5
    window_start = max(0, current_time - 0.5)

    if full_text.startswith(context["confirmed_text"]):
        new_blob = full_text[len(context["confirmed_text"]):].strip()
    else:
        new_blob = full_text
        context["word_timestamps"] = []

    if new_blob:
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9\']+', new_blob)
        if tokens:
            time_per = 0.5 / len(tokens)
            for idx, token in enumerate(tokens):
                context["word_timestamps"].append({
                    "word": token,
                    "start": round(window_start + (idx * time_per), 2),
                    "end": round(window_start + ((idx + 1) * time_per), 2)
                })
        context["confirmed_text"] = full_text

    return {
        "text": full_text,
        "words": context["word_timestamps"],
        "current_time_pos": current_time,
        "is_final": False
    }

@socket_router.websocket("/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    asr = websocket.app.state.asr_model
    
    # Initialize Qwen State
    state = asr.init_streaming_state(
        unfixed_chunk_num=4,
        unfixed_token_num=5,
        chunk_size_sec=0.5,
    )
    
    session_context = {
        "confirmed_text": "",
        "word_timestamps": []
    }
    
    try:
        while True:
            data = await websocket.receive_bytes()
            wav = np.frombuffer(data, dtype=np.float32).reshape(-1)
            asr.streaming_transcribe(wav, state)
            result = update_fixed_window_metadata_ws(state, session_context)
            
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        # Handle cleanup on disconnect
        asr.finish_streaming_transcribe(state)
        final_result = update_fixed_window_metadata_ws(state, session_context)
        final_result["is_final"] = True
        print("WebSocket disconnected. Session finished.")
        