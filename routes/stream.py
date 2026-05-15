import time
import uuid
import numpy as np
from typing import Dict
from dataclasses import dataclass, field
from fastapi import APIRouter, Request, HTTPException, Query, Depends
import re

@dataclass
class Session:
    state: object
    created_at: float
    last_seen: float
    confirmed_text: str = ""
    word_timestamps: list = field(default_factory=list)

stream_router = APIRouter(prefix="/stream", tags=["streaming-asr"])
SESSIONS: Dict[str, Session] = {}
SESSION_TTL_SEC = 10 * 60

def get_asr_model(request: Request):
    return request.app.state.asr_model

def gc_sessions(asr):
    now = time.time()
    dead = [sid for sid, s in SESSIONS.items() if now - s.last_seen > SESSION_TTL_SEC]
    for sid in dead:
        try:
            asr.finish_streaming_transcribe(SESSIONS[sid].state)
        except Exception: pass
        SESSIONS.pop(sid, None)

def update_fixed_window_metadata(session):
    state = session.state
    full_text = getattr(state, "text", "") or ""
    current_time = getattr(state, "chunk_id", 0) * 0.5
    window_start = max(0, current_time - 0.5)

    # 1. Identify only the NEW text added in this chunk
    if full_text.startswith(session.confirmed_text):
        new_text_blob = full_text[len(session.confirmed_text):].strip()
    else:
        # If the model corrected previous history, we wipe and start over
        new_text_blob = full_text
        session.word_timestamps = []

    if new_text_blob:
        # 2. Tokenize the new blob (Mix of Chinese characters and English words)
        # This regex treats Chinese chars as individual tokens and English words as whole tokens
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9\']+', new_text_blob)
        
        if tokens:
            # 3. Distribute the 0.5s window across the tokens
            # If 5 tokens appeared in this 0.5s chunk, each gets 0.1s
            time_per_token = 0.5 / len(tokens)
            
            for idx, token in enumerate(tokens):
                t_start = window_start + (idx * time_per_token)
                t_end = t_start + time_per_token
                
                session.word_timestamps.append({
                    "word": token,
                    "start": round(t_start, 2),
                    "end": round(t_end, 2)
                })
        
        session.confirmed_text = full_text

    return {
        "text": full_text,
        "language": getattr(state, "language", "") or "",
        "words": session.word_timestamps, # Key changed to 'words' to match your batch JSON
        "current_time_pos": current_time,
        "is_final": getattr(state, "is_final", False)
    }

@stream_router.post("/start")
async def api_start(request: Request, asr=Depends(get_asr_model)):
    gc_sessions(asr)
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
    
    # 1. Inference
    asr.streaming_transcribe(wav, s.state)
    
    # 2. Extract with Fixed Window Timestamping
    return update_fixed_window_metadata(s)

@stream_router.post("/finish")
async def api_finish(session_id: str = Query(...), asr=Depends(get_asr_model)):
    s = SESSIONS.pop(session_id, None)
    if not s:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    
    asr.finish_streaming_transcribe(s.state)
    return update_fixed_window_metadata(s)