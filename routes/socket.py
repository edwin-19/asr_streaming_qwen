import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import torch

socket_router = APIRouter(prefix="/ws", tags=["socket-streaming-asr"])

# Configuration boundaries
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 4  # float32 is 4 bytes
MAX_DURATION_SEC = 25  # 5 minutes strict limit for Qwen3 ForcedAligneer comfort zone
MAX_AUDIO_BYTES = MAX_DURATION_SEC * SAMPLE_RATE * BYTES_PER_SAMPLE

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
    
    audio_buffer = []
    total_bytes_received = 0
    limit_exceeded = False
    
    try:
        while True:
            data = await websocket.receive_bytes()
            total_bytes_received += len(data)
            
            # Memory Guardrail: Prevent RAM abuse
            if total_bytes_received > MAX_AUDIO_BYTES:
                print(f"Warning: Session exceeded max duration of {MAX_DURATION_SEC}s. Truncating buffer.")
                limit_exceeded = True
                # Option A: Break and force finalization right now
                break 
                
            wav = np.frombuffer(data, dtype=np.float32).reshape(-1)
            audio_buffer.append(wav)
            
            # Streaming Inference
            asr.streaming_transcribe(wav, state)
            
            await websocket.send_json({
                "text": getattr(state, "text", ""),
                "is_final": False
            })
            
    except WebSocketDisconnect:
        print("Client disconnected normally.")
        
    finally:
        # --- 2. Finalization & Forced Alignment ---
        asr.finish_streaming_transcribe(state)
        final_text = getattr(state, "text", "")
        
        final_payload = {
            "text": final_text,
            "words": [],
            "is_final": True,
            "truncated": limit_exceeded
        }

        if final_text and audio_buffer:
            # Create a localized block to easily scope out heavy objects for GC
            try:
                full_audio = np.concatenate(audio_buffer)
                
                # Double-check duration to avoid CUDA OOM inside Qwen3 Forced Aligner
                duration_sec = len(full_audio) / SAMPLE_RATE
                if duration_sec <= MAX_DURATION_SEC:
                    alignment_results = asr.forced_aligner.align(
                        audio=(full_audio, SAMPLE_RATE),
                        text=final_text,
                        language=getattr(state, "language", "en")
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
                    print(f"Audio too long ({duration_sec:.1f}s) for safe forced alignment. Skipping alignment step.")
                    
            except Exception as e:
                print(f"Forced alignment failed: {e}")
            finally:
                # Proactively clean heavy variables out of memory
                del audio_buffer
                if 'full_audio' in locals():
                    del full_audio
                if torch.cuda.is_available():
                    torch.cuda.empty_cache() # Clear VRAM fragmentation cache
        
        # Send the final chunk payload safely
        try:
            await websocket.send_json(final_payload)
            await websocket.close()
        except Exception:
            pass
            
        print(f"WS session cleaned up. Final text length: {len(final_text)}")