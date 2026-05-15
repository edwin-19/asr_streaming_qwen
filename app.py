from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routes.transcribe import transcribe_route
from routes.stream import stream_router
from routes.socket import socket_router
import torch
from qwen_asr import Qwen3ASRModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.asr_model = Qwen3ASRModel.LLM(
        'assets/qwen3-asr-finetuning-out-v3/checkpoint-727/', 
        # 'weights/qwen3-asr-finetuning-out-v2/checkpoint-2908', 
        max_inference_batch_size=2,
        gpu_memory_utilization=0.8,
        max_model_len=8192,
        forced_aligner="assets/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
            # attn_implementation="flash_attention_2",
        ),
    )
    
    app.state.unfixed_chunk_num = 4
    app.state.unfixed_token_num = 5
    app.state.chunk_size_sec = 1.0
    
    yield
    # SHUTDOWN: Clean up vLLM
    if hasattr(app.state.asr_model.model, "llm_engine"):
        app.state.asr_model.model.llm_engine.shutdown()
    torch.cuda.empty_cache()

origins = ["*"]
tags_metadata = [
    {"name": "transcribe", "description": "Calling the Qwen ASR Engine"},
]

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcribe_route)
app.include_router(stream_router)
app.include_router(socket_router)

@app.get('/health', tags=['Health Check'])
async def health():
    return {
        'status': 'ok'
    }