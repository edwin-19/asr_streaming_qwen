import json
import time
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

socket_router = APIRouter(prefix="/ws", tags=["socket-streaming-asr"])

@socket_router.websocket("/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    asr = websocket.app.state.asr_model
    
    # 1. Initialize ASR Streaming State
    state = asr.init_streaming_state(
        unfixed_chunk_num=4,
        unfixed_token_num=5,
        chunk_size_sec=0.5,
    )
    
    # Buffer to accumulate raw audio for the forced aligner
    audio_buffer = []
    
    try:
        while True:
            # Receive binary audio chunk
            data = await websocket.receive_bytes()
            wav = np.frombuffer(data, dtype=np.float32).reshape(-1)
            audio_buffer.append(wav)
            
            # Streaming Inference (updates state.text)
            asr.streaming_transcribe(wav, state)
            
            # Send partial text recognition to client
            await websocket.send_json({
                "text": getattr(state, "text", ""),
                "is_final": False
            })
            
    except WebSocketDisconnect:
        # --- 2. Finalization & Forced Alignment ---
        asr.finish_streaming_transcribe(state)
        final_text = getattr(state, "text", "")
        
        final_payload = {
            "text": final_text,
            "words": [],
            "is_final": True
        }

        if final_text and audio_buffer:
            try:
                # Combine all buffered audio for the aligner
                full_audio = np.concatenate(audio_buffer)
                
                # Perform the alignment
                alignment_results = asr.forced_aligner.align(
                    audio=(full_audio, 16000),
                    text=final_text,
                    language=getattr(state, "language", "en")
                )
                
                # Extract and map items to our standard format
                items = alignment_results[0].items if alignment_results else []
                final_payload["words"] = [
                    {
                        "word": item.text,
                        "start": item.start_time,
                        "end": item.end_time
                    } for item in items
                ]
            except Exception as e:
                print(f"Forced alignment failed: {e}")
        
        # Send the finalized, aligned data
        try:
            await websocket.send_json(final_payload)
        except Exception:
            # Socket might already be fully closed by the client
            pass
            
        print(f"WS session closed. Final text length: {len(final_text)}")