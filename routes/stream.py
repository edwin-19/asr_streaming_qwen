import time
import uuid
import io
import numpy as np
import soundfile as sf
import librosa
from typing import Dict
from dataclasses import dataclass, field
from fastapi import APIRouter, Request, HTTPException, Query, Depends
from fastapi.concurrency import run_in_threadpool

@dataclass
class Session:
    state: object
    created_at: float
    last_seen: float
    # Accumulate audio for alignment at the end
    audio_buffer: list = field(default_factory=list) 
    language: str = "en"

stream_router = APIRouter(prefix="/stream", tags=["streaming-asr"])
SESSIONS: Dict[str, Session] = {}
SESSION_TTL_SEC = 10 * 60

def get_asr_model(request: Request):
    return request.app.state.asr_model

@stream_router.post("/start")
async def api_start(request: Request, asr=Depends(get_asr_model)):
    session_id = uuid.uuid4().hex
    state = asr.init_streaming_state(
        unfixed_chunk_num=getattr(request.app.state, "unfixed_chunk_num", 4),
        unfixed_token_num=getattr(request.app.state, "unfixed_token_num", 5),
        chunk_size_sec=getattr(request.app.state, "chunk_size_sec", 1.0),
    )
    now = time.time()
    SESSIONS[session_id] = Session(state=state, created_at=now, last_seen=now)
    return {"session_id": session_id}

@stream_router.post("/chunk")
async def api_chunk(session_id: str = Query(...), request: Request = None, asr=Depends(get_asr_model)):
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    
    s.last_seen = time.time()
    raw = await request.body()
    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1)
    
    # Store audio for later alignment
    s.audio_buffer.append(wav)
    
    # Perform streaming inference
    # asr.streaming_transcribe(wav, s.state)
    await run_in_threadpool(asr.streaming_transcribe, wav, s.state)
    
    # Update language if detected during stream
    if hasattr(s.state, "language") and s.state.language:
        s.language = s.state.language

    return {
        "text": getattr(s.state, "text", "") or "",
        "is_final": False
    }

@stream_router.post("/finish")
async def api_finish(session_id: str = Query(...), asr=Depends(get_asr_model)):
    s = SESSIONS.pop(session_id, None)
    if not s:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    
    # 1. Finalize ASR state to get total text
    await run_in_threadpool(asr.finish_streaming_transcribe, s.state)
    final_text = getattr(s.state, "text", "")

    if not final_text or not s.audio_buffer:
        return {"text": final_text, "alignment": []}

    # 2. Reconstruct full audio from chunks
    full_audio = np.concatenate(s.audio_buffer)

    # 3. Run Forced Aligner
    # Note: Assuming asr_model has a forced_aligner attribute as per your snippet
    aligner = asr.forced_aligner
    
    try:
        # We assume the streaming chunks are already 16k float32 
        # based on the api_chunk logic.
        # alignment_results = aligner.align(
        #     audio=(full_audio, 16000), 
        #     text=final_text, 
        #     language=s.language
        # )
        alignment_results = await run_in_threadpool(
            aligner.align, 
            audio=(full_audio, 16000), 
            text=final_text, 
            language=s.language
        )
        # Handle case where aligner returns list of results
        results = alignment_results[0] if isinstance(alignment_results, list) else alignment_results
    except Exception as e:
        # Fallback if alignment fails (e.g. text/audio mismatch)
        results = {"error": f"Alignment failed: {str(e)}"}

    return {
        "text": final_text,
        "alignment": results
    }