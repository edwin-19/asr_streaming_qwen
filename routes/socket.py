import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import torch
import anyio  # FastAPI's background worker runner

socket_router = APIRouter(prefix="/ws", tags=["socket-streaming-asr"])

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 4  # float32 is 4 bytes
MAX_DURATION_SEC = 60  # Aligned to a safe 1-minute guardrail for vLLM/ForcedAligner
MAX_AUDIO_BYTES = MAX_DURATION_SEC * SAMPLE_RATE * BYTES_PER_SAMPLE

# --- BACKEND ROUTER UPDATE ---
@socket_router.websocket("/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    asr = websocket.app.state.asr_model
    
    state = asr.init_streaming_state(
        unfixed_chunk_num=4,
        unfixed_token_num=5,
        chunk_size_sec=0.5,
    )
    
    audio_buffer = []
    pending_bytes = b""
    total_bytes_received = 0
    limit_exceeded = False
    
    try:
        while True:
            # Check the incoming frame type dynamically
            message = await websocket.receive()
            
            # Handle Text control signals (like EOF)
            if "text" in message:
                text_data = message["text"]
                if text_data == "EOF":
                    print("Client signaled EOF. Breaking loop to run finalization...")
                    break
                continue
                
            # Handle Binary audio data frames
            if "bytes" in message:
                data = message["bytes"]
                if not data:
                    continue
                    
                total_bytes_received += len(data)
                if total_bytes_received > MAX_AUDIO_BYTES:
                    print(f"Warning: Session exceeded max duration. Truncating.")
                    limit_exceeded = True
                    break 
                
                pending_bytes += data
                remainder = len(pending_bytes) % BYTES_PER_SAMPLE
                if remainder == 0:
                    bytes_to_process = pending_bytes
                    pending_bytes = b""
                else:
                    bytes_to_process = pending_bytes[:-remainder]
                    pending_bytes = pending_bytes[-remainder:]
                    
                if not bytes_to_process:
                    continue

                wav = np.frombuffer(bytes_to_process, dtype=np.float32).reshape(-1)
                audio_buffer.append(wav)
                
                await anyio.to_thread.run_sync(asr.streaming_transcribe, wav, state)
                
                await websocket.send_json({
                    "text": getattr(state, "text", ""),
                    "is_final": False
                })
                
    except WebSocketDisconnect:
        print("Client disconnected abruptly.")
        
    finally:
        # Finalize transcription
        await anyio.to_thread.run_sync(asr.finish_streaming_transcribe, state)
        final_text = getattr(state, "text", "")
        
        final_payload = {
            "text": final_text,
            "words": [],
            "is_final": True,
            "truncated": limit_exceeded
        }

        if final_text and audio_buffer:
            try:
                full_audio = np.concatenate(audio_buffer)
                duration_sec = len(full_audio) / SAMPLE_RATE
                
                if duration_sec <= MAX_DURATION_SEC:
                    alignment_results = await anyio.to_thread.run_sync(
                        lambda: asr.forced_aligner.align(
                            audio=(full_audio, SAMPLE_RATE),
                            text=final_text,
                            language=getattr(state, "language", "en")
                        )
                    )
                    
                    items = alignment_results[0].items if alignment_results else []
                    final_payload["words"] = [
                        {
                            "word": item.text,
                            "start": item.start_time,
                            "end": item.end_time
                        } for item in items
                    ]
                else:
                    print(f"Audio too long ({duration_sec:.1f}s) for safe alignment.")
                    
            except Exception as e:
                print(f"Forced alignment failed: {e}")
            finally:
                del audio_buffer
                if 'full_audio' in locals():
                    del full_audio
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Send final payload down the open pipe, THEN let it naturally close
        try:
            await websocket.send_json(final_payload)
            print("Successfully sent final alignment payload to client.")
        except Exception as e:
            print(f"Failed to transmit final payload: {e}")
            
        print(f"WS session cleaned up cleanly.")