from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routes.transcribe import transcribe_route
from routes.stream import stream_router
from routes.socket import socket_router
import torch
from qwen_asr import Qwen3ASRModel
from fastapi.responses import HTMLResponse
import os
import anyio

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("ASR_MODEL_PATH", "weights/qwen3-asr-finetuning-out-v2/checkpoint-2908")
    forced_aligner_path = os.getenv("FORCED_ALIGNER_PATH", "assets/Qwen3-ForcedAligner-0.6B")
    
    max_batch_size = int(os.getenv("ASR_MAX_BATCH_SIZE", "2"))
    gpu_utilization = float(os.getenv("ASR_GPU_MEMORY_UTILIZATION", "0.9"))
    max_model_len = int(os.getenv("ASR_MAX_MODEL_LEN", "8192"))
    device = os.getenv("ASR_DEVICE", "cuda:0")
    
    app.state.asr_model = Qwen3ASRModel.LLM(
        model_path, 
        max_inference_batch_size=max_batch_size,
        gpu_memory_utilization=gpu_utilization,
        max_model_len=max_model_len,
        forced_aligner=forced_aligner_path,
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map=device,
            # attn_implementation="flash_attention_2", # Optional: can also toggle via env if needed
        ),
    )
    
    app.state.unfixed_chunk_num = int(os.getenv("ASR_UNFIXED_CHUNK_NUM", "4"))
    app.state.unfixed_token_num = int(os.getenv("ASR_UNFIXED_TOKEN_NUM", "5"))
    app.state.chunk_size_sec = float(os.getenv("ASR_CHUNK_SIZE_SEC", "1.0"))
    
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_threads = 4
    
    yield
    # SHUTDOWN: Clean up vLLM
    if hasattr(app.state.asr_model.model, "llm_engine"):
        app.state.asr_model.model.llm_engine.shutdown()
    torch.cuda.empty_cache()

tags_metadata = [
    {"name": "transcribe", "description": "Calling the Qwen ASR Engine"},
    {"name": "streaming-asr", "description": "Streaming Endpoint Functionality"},
    {"name": "socket-streaming-asr", "description": "Socket ASR Endpoint"},
]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    
@app.get('/', response_class=HTMLResponse)
async def get_ui():
    html_path = os.path.join(TEMPLATE_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())