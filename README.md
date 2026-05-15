# Streaming ASR code swithcing
A high-performance, streaming Automatic Speech Recognition system fine-tuned for English-Mandarin code-switching using the [SEAME](https://www.isca-archive.org/interspeech_2010/lyu10_interspeech.pdf) (Singapore-Malaysia) dataset.


# Prerequites
- Install necessary libs
```
pip install -r requirments.txt
```

- Download test dataset
```bash
hf download AudioLLMs/seame_dev_sge \
  --repo-type dataset \
  --local-dir ./data/seame_dev_sge
```

# Training
- Training use the qwen3_asr_sft.py of 
```
bash train.sh
```

# Local Inference
```bash
# Run and get CER and WER
python eval.py

# Get timestamp
python get_timestamp.py
```

# Startup fastapi server
```bash
uvicorn app:app --host 0.0.0.0
```

- Run inference on server
```bash
# Run main
python req_trans.py main

# Run streaming inference
python req_trans.py run-stream

# Run websocket
python req_trans.py run-ws-stream
```

# Build docker env
```bash
docker compose up -d --build
```

# ASR Evaluation Report: Qwen Models on SEAME Dataset

## 1. Summary of Results
| Model Variant | Samples | WER (Word Error Rate) | CER (Character Error Rate) |
| :--- | :---: | :---: | :---: |
| Qwen 0.6B Pretrained | 3003 | 42.60% | 30.10% |
| **Qwen 0.6B FT SEAME** | 3003 | **21.93%** | **15.12%** |
| Qwen 1.7B Pretrained | 3003 | 37.36% | 25.02% |
| **Qwen 1.7B FT SEAME** | 3003 | **21.00%** | **14.54%** |

## 2. Comparative Analysis
* **The 0.6B Advantage:** The 0.6B model achieves performance remarkably close to the 1.7B variant (only ~0.93% difference).
* **Fine-tuning Impact:** Fine-tuning resulted in a **48.52%** relative error reduction for the 0.6B model.
* **Mandarin Impact:** Mandarin prefers the usage of CER which is better than using WER but i added in just in case