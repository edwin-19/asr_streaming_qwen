import typer
import httpx
import async_typer
import random
import io
import time
import soundfile as sf
from datasets import load_dataset
from jiwer import wer, cer
import re
import time
import numpy as np
import json
import websockets
import asyncio

app = async_typer.AsyncTyper()

def normalize_mixed_text(text: str) -> str:
    if not text: return ""
    text = text.lower()
    
    # 1. Remove filler tags like (uh), [ah], <unk>
    text = re.sub(r'[\(\[\<].*?[\)\]\>]', '', text)
    
    # 2. Handle the "Space" issue: Put spaces around Chinese characters
    # This ensures "他跟" becomes "他 跟" so they are counted as individual tokens
    text = re.sub(r'([\u4e00-\u9fff])', r' \1 ', text)
    
    # 3. Clean up punctuation and extra whitespace
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    text = " ".join(text.split())
    
    return text

@app.async_command()
async def main(
    dataset_path: str = typer.Option("./data/seame_dev_sge"),
    api_url: str = typer.Option("http://localhost:8000/v1/audio/predict"),
    num_samples: int = typer.Option(10) # Default stays at 10
):
    dataset = load_dataset(dataset_path)['test']
    
    # --- FIX: Only select 10 samples ---
    # Ensure we don't try to sample more than what exists in the dataset
    actual_num = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), actual_num)
    samples = dataset.select(indices)
    
    metrics = {"rtf": [], "wer": [], "cer": []}
    
    async with httpx.AsyncClient() as client:
        for i, example in enumerate(samples):
            audio_data = example.get("audio") or example.get("context")
            audio_array = audio_data["array"]
            sampling_rate = audio_data["sampling_rate"]
            duration = len(audio_array) / sampling_rate
            
            # 1. Get Ground Truth
            gt_raw = example.get("answer") or example.get("text") or ""
            gt_norm = normalize_mixed_text(gt_raw)
            
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sampling_rate, format='WAV')
            buffer.seek(0)
            
            files = {"file": ("test.wav", buffer, "audio/wav")}
            
            try:
                start_time = time.perf_counter()
                response = await client.post(api_url, files=files, timeout=60.0)
                response.raise_for_status()
                inference_time = time.perf_counter() - start_time
                
                # --- NEW: SHOW WHOLE JSON ---
                result = response.json()
                print(f"\n[DEBUG] Raw JSON Response Sample {i+1}:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
                hyp_raw = result.get('text', "")
                hyp_norm = normalize_mixed_text(hyp_raw)
                
                # 2. Calculate Metrics
                current_rtf = inference_time / duration
                current_wer = wer(gt_norm, hyp_norm) if gt_norm else 0
                current_cer = cer(gt_norm, hyp_norm) if gt_norm else 0
                
                metrics["rtf"].append(current_rtf)
                metrics["wer"].append(current_wer)
                metrics["cer"].append(current_cer)
                
                # --- LOGGING ---
                print(f"GT  : {gt_norm}")
                print(f"HYP : {hyp_norm}")
                print(f"WER : {current_wer:.2%} | CER: {current_cer:.2%} | RTF: {current_rtf:.3f}")
                print("-" * 30)
                
            except Exception as e:
                print(f"❌ Error on sample {i}: {e}")

    # 3. Final Summary Table
    if metrics["rtf"]:
        avg_wer = sum(metrics["wer"]) / len(metrics["wer"])
        avg_cer = sum(metrics["cer"]) / len(metrics["cer"])
        avg_rtf = sum(metrics["rtf"]) / len(metrics["rtf"])
        
        print("\n" + "="*40)
        print("FINAL EVALUATION SUMMARY")
        print("="*40)
        print(f"Total Samples: {len(metrics['rtf'])}")
        print(f"Average WER   : {avg_wer:.2%}")
        print(f"Average CER   : {avg_cer:.2%}")
        print(f"Average RTF   : {avg_rtf:.3f}")
        print(f"Throughput    : {1/avg_rtf:.2f}x Real-time")
        print("="*40)

@app.async_command()
async def run_stream(
    dataset_path: str = typer.Option("./data/seame_dev_sge"),
    base_url: str = typer.Option("http://localhost:8000/stream"),
    num_samples: int = typer.Option(2)
):
    # 1. Load and Prepare Data
    # cast_column ensures we aren't sending weird sample rates to the server
    ds = load_dataset(dataset_path)['test']
    
    stacked_audio = []
    ground_truths = []
    
    for i in range(min(num_samples, len(ds))):
        example = ds[i]
        stacked_audio.append(example["context"]["array"].astype(np.float32))
        raw_gt = example.get("answer") or example.get("text") or ""
        ground_truths.append(normalize_mixed_text(raw_gt))
    
    full_audio = np.concatenate(stacked_audio)
    full_gt = " ".join(ground_truths).strip()
    
    # --- 2. SETUP ---
    SAMPLE_RATE = 16000
    CHUNK_DURATION = 0.5 
    SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION)
    total_chunks = len(full_audio) // SAMPLES_PER_CHUNK
    latencies = []

    print(f"🚀 Starting Word-Level Stream: {len(full_audio)/SAMPLE_RATE:.2f}s audio")
    print(f"📦 Total Chunks: {total_chunks} | Chunk size: {CHUNK_DURATION}s")
    print("-" * 60)

    # --- 3. STREAMING ---
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        try:
            session_id = client.post("/start").json()["session_id"]
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return

        for i in range(total_chunks):
            start_idx = i * SAMPLES_PER_CHUNK
            chunk = full_audio[start_idx : start_idx + SAMPLES_PER_CHUNK]
            
            req_start = time.perf_counter()
            try:
                resp = client.post(
                    "/chunk", 
                    params={"session_id": session_id}, 
                    content=chunk.tobytes(), 
                    headers={"Content-Type": "application/octet-stream"}
                )
                latencies.append(time.perf_counter() - req_start)
                
                data = resp.json()
                # POLISH: Updated key to 'words' to match server-side interpolation
                words = data.get("words", [])
                
                # Dynamic terminal display
                if words:
                    # Show the 3 most recent words appearing in the stream
                    recent = words[-3:]
                    # Using 'word' and 'start' keys from the new JSON format
                    display = " | ".join([f"{w['word']} ({w['start']:.1f}s)" for w in recent])
                    print(f"\r[{i+1:03d}/{total_chunks}] {latencies[-1]*1000:4.0f}ms >> {display:<50}", end="", flush=True)
                else:
                    print(f"\r[{i+1:03d}/{total_chunks}] {latencies[-1]*1000:4.0f}ms >> ...{' ':<50}", end="", flush=True)
                
            except Exception as e:
                print(f"\n⚠️ Stream interrupted at chunk {i}: {e}")
                break
            
            # Real-time simulation: Sleep if processing was faster than chunk duration
            time.sleep(max(0, CHUNK_DURATION - (time.perf_counter() - req_start)))

        # --- 4. FINALIZATION & METRICS ---
        print("\n" + "="*60)
        final_resp = client.post("/finish", params={"session_id": session_id})
        final_data = final_resp.json()
        
        # POLISH: Pretty-print the raw JSON for research audit
        print(f"🔍 RAW JSON AUDIT:\n{json.dumps(final_data, indent=2, ensure_ascii=False)}")
        print("-" * 60)

        final_pred = normalize_mixed_text(final_data.get("text", "")).strip()
        final_words = final_data.get("words", [])
        
        audio_dur = len(full_audio) / SAMPLE_RATE
        rtf = sum(latencies) / audio_dur
        
        # Accuracy Calculation
        try:
            error_rate = cer([full_gt], [final_pred]) if full_gt and final_pred else 1.0
        except Exception:
            error_rate = -1.0

        print(f"📊 PERFORMANCE REPORT")
        print(f"RTF:    {rtf:.4f} ({'FAST' if rtf < 1 else 'SLOW'})")
        print(f"CER:    {error_rate:.2%}")
        print("-" * 60)
        
        print("🕒 WORD-LEVEL TIMELINE (Binned to 0.5s windows):")
        for w in final_words:
            print(f"  [{w['start']:5.2f}s - {w['end']:5.2f}s] : {w['word']}")
            
        print("-" * 60)
        print(f"PRED: {final_pred}")
        print(f"GT:   {full_gt}")
        print("="*60)

@app.async_command()
async def align_text(
    dataset_path: str = typer.Option("./data/seame_dev_sge"),
    api_url: str = typer.Option("http://localhost:8000/v1/audio/pred_timestamp"),
):
    dataset = load_dataset(dataset_path)['test']
    example = dataset[0]
    
    audio_data = example.get("audio") or example.get("context")
    audio_array = audio_data["array"]
    sampling_rate = audio_data["sampling_rate"]
    
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sampling_rate, format='WAV')
    buffer.seek(0)
    
    files = {"file": ("test.wav", buffer, "audio/wav")}
    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, files=files, params={'text': example['answer'], 'language': "Chinese"}, timeout=60.0)
        print(response.json())

@app.async_command()
async def run_ws_stream(
    dataset_path: str = typer.Option("./data/seame_dev_sge"),
    ws_url: str = typer.Option("ws://localhost:8000/ws/transcribe"),
    num_samples: int = typer.Option(10)
):
    # 1. Load and Prepare Stacked Audio
    # Ensure consistent 16kHz for timing math
    ds = load_dataset(dataset_path)['test']
    
    indices = random.sample(range(len(ds)), min(num_samples, len(ds)))
    samples = ds.select(indices)
    
    all_audio = []
    all_gt = []
    for feat in samples:
        all_audio.append(feat["context"]["array"].astype(np.float32))
        # Handle field naming variations in SEAME
        raw_gt = feat.get("answer") or feat.get("text") or ""
        all_gt.append(normalize_mixed_text(raw_gt))
        
    full_audio = np.concatenate(all_audio)
    full_gt = " ".join(all_gt).strip()
    
    # --- Configuration ---
    SAMPLE_RATE = 16000
    CHUNK_SEC = 0.5
    SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_SEC)
    total_chunks = len(full_audio) // SAMPLES_PER_CHUNK
    total_audio_dur = len(full_audio) / SAMPLE_RATE
    
    print(f"Streaming {num_samples} samples | Total Audio: {total_audio_dur:.2f}s")
    print("-" * 60)

    async with websockets.connect(ws_url) as websocket:
        latencies = []
        start_wall_clock = time.perf_counter()
        
        for i in range(total_chunks):
            loop_start = time.perf_counter()
            
            # Audio Pointers
            start_idx = i * SAMPLES_PER_CHUNK
            chunk = full_audio[start_idx : start_idx + SAMPLES_PER_CHUNK]
            current_audio_time = (i + 1) * CHUNK_SEC
            
            # Send/Recv
            await websocket.send(chunk.tobytes())
            response = await websocket.recv()
            
            latencies.append(time.perf_counter() - loop_start)
            data = json.loads(response)
            
            # Real-time Visuals
            words = data.get("words", [])
            recent_txt = " ".join([w['word'] for w in words[-3:]]) # Show last 3 words
            percent = (current_audio_time / total_audio_dur) * 100
            
            print(f"\r[{percent:3.0f}%] {current_audio_time:5.1f}s | {latencies[-1]*1000:4.0f}ms >> {recent_txt:<30}", end="", flush=True)
            
            # Real-time simulation (throttle to 0.5s chunks)
            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0, CHUNK_SEC - elapsed))

        # --- Final Metadata and Scoring ---
        total_wall_time = time.perf_counter() - start_wall_clock
        final_pred = normalize_mixed_text(data.get('text', ''))
        final_words = data.get("words", [])
        
        # Calculate CER
        error_rate = cer([full_gt], [final_pred]) if full_gt else 0.0

        print("\n" + "="*60)
        print(f"STACKED STREAM SUMMARY")
        print("-" * 60)
        print(f"Throughput RTF : {sum(latencies) / total_audio_dur:.4f}")
        print(f"Accuracy CER   : {error_rate:.2%}")
        print(f"Total Latency  : {sum(latencies):.2f}s for {total_audio_dur:.2f}s audio")
        print("-" * 60)
        
        # Optional: Print first/last few words with timestamps for verification
        if final_words:
            print("🕒 TIMELINE PREVIEW (First 5 tokens):")
            for w in final_words[:5]:
                print(f"  [{w['start']:5.2f}s - {w['end']:5.2f}s] : {w['word']}")
        
        print("-" * 60)
        print(f"GT:   {full_gt[:100]}...") # Truncated for display
        print(f"PRED: {final_pred[:100]}...")
        print("="*60)

if __name__ == "__main__":
    app()