import time
import numpy as np
from locust import HttpUser, SequentialTaskSet, task, between, events
from datasets import load_dataset
import random

# --- CONFIGURATION ---
TARGET_SR = 16000
CHUNK_SIZE_SEC = 1.0
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SIZE_SEC)

# Global pool to hold pre-sliced audio chunks in memory
AUDIO_POOL = []

@events.init.add_listener
def on_locust_init(environment, **kwargs):
    try:
        ds = load_dataset("./data/seame_dev_sge")['test']
        
        for item in ds:
            array = item["context"]["array"].astype(np.float32)
            # text_truth = item["answer"]
            
            file_chunks = []
            for i in range(0, len(array), CHUNK_SAMPLES):
                chunk = array[i:i + CHUNK_SAMPLES]
                if len(chunk) < CHUNK_SAMPLES:
                    chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)), 'constant')
                file_chunks.append(chunk.tobytes())
                
            if file_chunks:
                AUDIO_POOL.append(file_chunks[:10])
            
        print(f"Successfully pooled {len(AUDIO_POOL)} audio files from SEAME.")
    except Exception as e:
        print(f"Error preloading dataset: {e}")
        # Fallback to dummy data so Locust doesn't crash completely
        print("Falling back to dummy noise generation...")
        dummy_file = [np.random.uniform(-0.1, 0.1, CHUNK_SAMPLES).astype(np.float32).tobytes() for _ in range(5)]
        AUDIO_POOL.append(dummy_file)

class StreamingASRWorkflow(SequentialTaskSet):
    def on_start(self):
        self.session_id = None
        self.my_chunks = random.choice(AUDIO_POOL)
    
    @task
    def start_session(self):
        with self.client.post('/stream/start', catch_response=True) as response:
            if response.status_code == 200:
                try:
                    self.session_id = response.json().get("session_id")
                    if not self.session_id:
                        response.failure("Response missing session_id")
                except Exception as e:
                    response.failure(f"Failed to parse JSON: {e}")
            else:
                response.failure(f"Failed to start session: {response.status_code}")
                
    @task
    def stream_chunks(self):
        if not self.session_id:
            self.interrupt()
            return
        
        for i , chunk_bytes in enumerate(self.my_chunks):
            if i > 0:
                time.sleep(CHUNK_SIZE_SEC)
            
            headers = {"Content-Type": "application/octet-stream"}
            url = f"/stream/chunk?session_id={self.session_id}"
            
            with self.client.post(url, data=chunk_bytes, headers=headers, name="/stream/chunk", catch_response=True) as response:
                if response.status_code != 200:
                    response.failure(f"Chunk {i} failed: {response.status_code}")
                    break
    
    @task
    def finish_session(self):
        if not self.session_id:
            self.interrupt()
            return
        
        url = f"/stream/finish?session_id={self.session_id}"
        with self.client.post(url, name='/stream/finish', catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "alignment" in data and isinstance(data["alignment"], dict) and "error" in data["alignment"]:
                    response.failure(f"ASR finished but Aligner failed: {data['alignment']['error']}")
            else:
                response.failure(f"Failed to finish session: {response.status_code}")
        
        self.interrupt()
        
class ASRBenchUser(HttpUser):
    tasks = [StreamingASRWorkflow]
    wait_time = between(0.5, 2.0)
                    
                    