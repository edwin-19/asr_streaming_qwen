import typer
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
import torch
from tqdm import tqdm
import jiwer
import json
from pathlib import Path
from req_trans import normalize_mixed_text

app = typer.Typer()

def calculate_metrics(refs, preds):
    """Calculates both WER and CER for code-switching data."""
    # Word Error Rate
    wer = jiwer.wer(refs, preds)
    
    # Character Error Rate (Crucial for Mandarin components)
    # CER treats every character (including Chinese characters) as a unit
    cer = jiwer.cer(refs, preds)
    
    return wer, cer

@app.command()
def eval(
    dataset_path: str = typer.Option("./data/seame_dev_sge"),
    model_path: str = typer.Option("./models/Qwen3-ASR-0.6B"),
    device: str = typer.Option("cuda"),
    output_dir: str = typer.Option("./results"),
    batch_size: int = typer.Option(16),
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_path)['test']

    print("Sorting dataset by audio length to minimize padding...")
    indexed_data = []
    for i, d in enumerate(dataset):
        indexed_data.append({
            "idx": i,
            "length": len(d['context']['array']),
            "data": d
        })
    indexed_data.sort(key=lambda x: x['length'])

    model = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=batch_size, # This controls internal batching
    )

    all_pred, all_gt, logs = [], [], []

    for i in tqdm(range(0, len(indexed_data), batch_size)):
        batch_items = indexed_data[i : i + batch_size]
        
        # Extract audio and ground truths
        audios = [
            (item['data']['context']['array'], item['data']['context']['sampling_rate']) 
            for item in batch_items
        ]
        batch_gt_raw = [item['data'].get('answer', "") for item in batch_items]

        try:
            # model.transcribe accepts a list of tuples for batching
            results_list = model.transcribe(audio=audios)
            
            for j, results in enumerate(results_list):
                norm_gt = normalize_mixed_text(batch_gt_raw[j])
                if not norm_gt.strip():
                    continue
                
                norm_pred = normalize_mixed_text(results.text or "")
                
                all_gt.append(norm_gt)
                all_pred.append(norm_pred)
                logs.append({
                    "id": batch_items[j]['idx'],
                    "ref": norm_gt,
                    "pred": norm_pred,
                    "wer": jiwer.wer(norm_gt, norm_pred)
                })

        except Exception as e:
            print(f"Error processing batch at index {i}: {e}")
            
    if not all_gt:
        print("No valid data processed.")
        return

    wer, cer = calculate_metrics(all_gt, all_pred)

    print(f"\n--- Evaluation Results ---")
    print(f"Total Samples: {len(all_gt)}")
    print(f"WER: {wer:.4f} ({wer*100:.2f}%)")
    print(f"CER: {cer:.4f} ({cer*100:.2f}%)")
    
    # Save results to disk
    result_file = Path(output_dir) / "eval_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({"wer": wer, "cer": cer, "samples": logs}, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    app()