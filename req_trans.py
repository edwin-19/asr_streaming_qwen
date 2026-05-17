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
    
    actual_num = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), actual_num)
    samples = dataset.select(indices)
    
    metrics = {"rtf": [], "wer": [], "cer": []}
    
    async with httpx.AsyncClient(timeout=60) as client:
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
                print(f"Error on sample {i}: {e}")

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
    num_samples: int = typer.Option(7), 
    chunk_size: float = typer.Option(0.5)
):
    ds = load_dataset(dataset_path)['test']
    
    stacked_audio = []
    ground_truths = []
    
    for i in range(min(num_samples, len(ds))):
        example = ds[i]
        audio_data = example["context"]["array"].astype(np.float32)
        stacked_audio.append(audio_data)
        raw_gt = example.get("answer") or ""
        ground_truths.append(raw_gt)
    
    full_audio = np.concatenate(stacked_audio)
    full_gt = " ".join(ground_truths).strip()
    full_gt = normalize_mixed_text(full_gt)
    
    SAMPLE_RATE = 16000
    SAMPLES_PER_CHUNK = int(SAMPLE_RATE * chunk_size)
    total_chunks = len(full_audio) // SAMPLES_PER_CHUNK
    latencies = []

    print(f" Starting Stream: {len(full_audio)/SAMPLE_RATE:.2f}s audio")
    print(f" Chunks: {total_chunks} | Size: {chunk_size}s")
    print("-" * 60)

    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
        try:
            # Initialize Session
            result = await client.post("/start")
            session_id = result.json()["session_id"]
        except Exception as e:
            print(f" Connection Error: {e}")
            return

        for i in range(total_chunks):
            start_idx = i * SAMPLES_PER_CHUNK
            chunk = full_audio[start_idx : start_idx + SAMPLES_PER_CHUNK]
            
            req_start = time.perf_counter()
            try:
                chunk_bytes = bytes(chunk.tobytes())
                
                resp = await client.post(
                    "/chunk", 
                    params={"session_id": session_id}, 
                    content=chunk_bytes, 
                    headers={"Content-Type": "application/octet-stream"}
                )
                latencies.append(time.perf_counter() - req_start)
                
                data = resp.json()
                current_text = data.get("text", "")
                
                print(f"\r[{i+1:03d}/{total_chunks}] {latencies[-1]*1000:4.0f}ms >> {current_text[-50:]:<50}", end="", flush=True)
                
            except Exception as e:
                print(f"\nStream interrupted at chunk {i}: {e}")
                break
            
            elapsed = time.perf_counter() - req_start
            
            # Only sleep if processing was faster than the audio chunk duration
            if elapsed < chunk_size:
                await asyncio.sleep(chunk_size - elapsed)

        print("\n" + "="*60)
        print("Finalizing stream and running forced aligner")
        
        final_resp = await client.post("/finish", params={"session_id": session_id})
        final_data = final_resp.json()
        
        # Extract fields based on your specific JSON response
        final_pred = normalize_mixed_text(final_data.get("text", "").strip())
        # Accessing the nested 'items' list
        alignment_data = final_data.get("alignment", {})
        word_items = alignment_data.get("items", [])
        
        audio_dur = len(full_audio) / SAMPLE_RATE
        rtf = sum(latencies) / audio_dur
        
        # Accuracy Calculation (CER)
        try:
            error_rate = cer([full_gt], [final_pred]) if full_gt and final_pred else 1.0
        except Exception:
            error_rate = -1.0

        print(f"Performance Report")
        print(f"RTF:    {rtf:.4f} ({'FAST' if rtf < 1 else 'SLOW'})")
        print(f"CER:    {error_rate:.2%}")
        print("-" * 60)
        
        print(" Word Level Timestamp :")
        # Displaying first 30 tokens/words
        for item in word_items[:30]:
            t_text = item.get("text", "???")
            t_start = item.get("start_time", 0.0)
            t_end = item.get("end_time", 0.0)
            
            # Formatting: align text to the left, timestamps to the right
            print(f"  [{t_start:5.2f}s -> {t_end:5.2f}s] : {t_text}")
        
        if len(word_items) > 30:
            print(f"  ... and {len(word_items) - 30} more tokens.")
            
        print("-" * 60)
        print(f"Pred: \t{final_pred}")
        print(f"Full GT:   {full_gt}")
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
        for item in response.json()['items']:
            print(item)

@app.async_command()
async def run_ws_stream(
    dataset_path: str = typer.Option("./data/seame_dev_sge"),
    ws_url: str = typer.Option("ws://localhost:8000/ws/transcribe"),
    num_samples: int = typer.Option(5),
    chunk_size: float = typer.Option(0.5)
):
    ds = load_dataset(dataset_path)['test']
    indices = random.sample(range(len(ds)), min(num_samples, len(ds)))
    samples = ds.select(indices)
    
    all_audio = []
    all_gt = []
    for feat in samples:
        all_audio.append(feat["context"]["array"].astype(np.float32))
        raw_gt = feat.get("answer") or ""
        all_gt.append(normalize_mixed_text(raw_gt))
        
    full_audio = np.concatenate(all_audio)
    full_gt = " ".join(all_gt).strip()
    
    SAMPLE_RATE = 16000
    SAMPLES_PER_CHUNK = int(SAMPLE_RATE * chunk_size)
    total_chunks = len(full_audio) // SAMPLES_PER_CHUNK
    total_audio_dur = len(full_audio) / SAMPLE_RATE
    
    print(f" Streaming {num_samples} samples | Total Audio: {total_audio_dur:.2f}s")
    print("-" * 60)

    async with websockets.connect(ws_url) as websocket:
        latencies = []

        for i in range(total_chunks):
            loop_start = time.perf_counter()
            
            start_idx = i * SAMPLES_PER_CHUNK
            chunk = full_audio[start_idx : start_idx + SAMPLES_PER_CHUNK]
            
            # Send binary audio chunk
            await websocket.send(chunk.tobytes())
            
            # Receive partial real-time streaming text response
            response = await websocket.recv()
            last_data = json.loads(response)
            
            latencies.append(time.perf_counter() - loop_start)
            
            current_text = last_data.get("text", "")
            percent = ((i + 1) * chunk_size / total_audio_dur) * 100
            print(f"\r[{percent:3.0f}%] >> {current_text[-50:]:<50}", end="", flush=True)
            
            # Real-time simulation spacing
            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0, chunk_size - elapsed))
        
        print("\n" + "="*60)
        print("Finishing audio input stream. Requesting Timestamp Results")
        
        # 1. Send the string signal telling the server we are done sending audio
        await websocket.send("EOF") 
        
        # 2. Wait patiently on the open connection for the heavy computation response frame
        final_data = {}
        try:
            final_response = await websocket.recv()
            final_data = json.loads(final_response)
        except Exception as e:
            print(f"Error receiving final alignment data: {e}")

        # 3. Process the results cleanly
        final_pred = normalize_mixed_text(final_data.get('text', ''))
        error_rate = cer([full_gt], [final_pred]) if full_gt else 0.0
        word_items = final_data.get("words", [])
        
        if word_items:
            print(f"\nWord Level Timestamp ({len(word_items)} tokens):")
            print("-" * 60)
            for item in word_items[:40]:  # Display up to 40 tokens cleanly
                w = item.get("word", "???")
                s = item.get("start", 0.0)
                e = item.get("end", 0.0)
                
                duration = e - s
                bar = "█" * int(duration * 10)  # 1 block per 100ms
                print(f"  [{s:6.2f}s -> {e:6.2f}s] | {w:<16} {bar}")
                
            if len(word_items) > 40:
                print(f"  ... and {len(word_items) - 40} more tokens.")
        else:
            print("\nNo alignment timeline data returned in final packet.")

        print("-" * 60)
        print(f"Websocket stream summary")
        print(f"RTF:    {sum(latencies) / total_audio_dur:.4f}")
        print(f"CER:    {error_rate:.2%}")
        if final_data.get("truncated"):
            print("Note: Server truncated audio due to maximum duration bounds.")
        print("-" * 60)
        print(f"GT:   {full_gt[:120]}...")
        print(f"PRED: {final_pred[:120]}...")
        print("="*60)

if __name__ == "__main__":
    app()