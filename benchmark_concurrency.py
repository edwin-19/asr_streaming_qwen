import typer
from async_typer import AsyncTyper

from datasets import load_dataset
import numpy as np
import websockets
import time
import json
import asyncio

app = AsyncTyper()

CHUNK_SIZE_SEC = 1.0
TARGET_SR = 16000 

# Calculate chunk size in samples
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SIZE_SEC)
WS_URL = "ws://localhost:8000/ws/transcribe"

def prepare_audio_chunks(audio_item):
    """Resamples and splits a HF dataset audio row into 16-bit PCM binary chunks."""
    array = audio_item["array"]
    orig_sr = audio_item["sampling_rate"]
    
    if array.dtype != np.int16:
        array = np.clip(array, -1.0, 1.0)
        pcm_array = (array * 32767).astype(np.int16)
    else:
        pcm_array = array
        
    chunks = []
    for i in range(0, len(pcm_array), CHUNK_SAMPLES):
        slice_data = pcm_array[i : i + CHUNK_SAMPLES]
        # Pad the last chunk with zeros if it's shorter than CHUNK_SAMPLES
        if len(slice_data) < CHUNK_SAMPLES:
            slice_data = np.pad(slice_data, (0, CHUNK_SAMPLES - len(slice_data)), 'constant')
        chunks.append(slice_data.tobytes())
        
    return chunks

async def simulate_ws_user(user_id: int, audio_item):
    chunks = prepare_audio_chunks(audio_item)
    print(f"[User {user_id}] Extracted {len(chunks)} chunks from dataset track. Connecting...")
    
    latencies = []
    
    try:
        async with websockets.connect(WS_URL) as ws:
            for idx, chunk in enumerate(chunks, start=1):
                start_time = time.perf_counter()
                
                await ws.send(chunk)
                
                response = await ws.recv()
                end_time = time.perf_counter()
                latency = end_time - start_time
                latencies.append(latency)
                
                try:
                    data = json.loads(response)
                    text_snippet = data.get("text", response)[:30]
                except json.JSONDecodeError:
                    text_snippet = str(response)[:30]
                print(f"[User {user_id}] Chunk {idx}/{len(chunks)} processed in {latency:.3f}s | Result: {text_snippet}...")
                
                await asyncio.sleep(CHUNK_SIZE_SEC)
    except Exception as e:
        print(f"[User {user_id}] WebSocket session failed: {e}")
        
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        rtf = avg_latency / CHUNK_SIZE_SEC
        print(f"\n--- [User {user_id} Results] ---")
        print(f"Average Server Response Time: {avg_latency:.3f}s")
        print(f"Real-Time Factor (RTF): {rtf:.3f}")

@app.async_command()
async def main(
    dataset_path: str = typer.Option("./data/seame_dev_sge"),
    ws_url: str = typer.Option("ws://localhost:8000/ws/transcribe"),
    users:int = typer.Option(1)
):
    DATASET_PATH = dataset_path  # Replace with your actual Hugging Face dataset path
    
    CONCURRENT_USERS = users
    
    print(f"Loading test split from dataset: {DATASET_PATH}...")
    ds = load_dataset(DATASET_PATH)['test']
    
    print(f"Initializing evaluation with {CONCURRENT_USERS} parallel WebSocket users...")
    tasks = []
    
    for i in range(CONCURRENT_USERS):
        audio_sample = ds[i % len(ds)]['context']
        tasks.append(simulate_ws_user(
            user_id=i + 1, audio_item=audio_sample
        ))
        
        await asyncio.sleep(0.1)
        
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    app()