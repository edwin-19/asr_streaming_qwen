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
    aligner_model_path: str = typer.Option("./models/Qwen3-ForcedAligner-0.6B"),
    device: str = typer.Option("cuda"),
    output_dir: str = typer.Option("./results"),
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset(dataset_path)['test']
    # dataset = dataset.select(range(100))
    
    model = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=32,
        max_new_tokens=256,
        # forced_aligner=aligner_model_path,
        # forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map=device),
    )

    all_pred, all_gt, logs = [], [], []

    for i, data in enumerate(tqdm(dataset)):
        # 1. Ground Truth Handling
        gt_text = data.get('answer', "")
        norm_gt = normalize_mixed_text(gt_text)
        
        if not norm_gt.strip():
            continue 
            
        # 2. Transcription with Error Catching
        try:
            audio = data['context']
            results = model.transcribe(
                audio=(audio['array'], audio['sampling_rate']), 
                # return_time_stamps=True
            )[0]
            
            raw_pred = results.text if results.text is not None else ""
            norm_pred = normalize_mixed_text(raw_pred)
            
            all_gt.append(norm_gt)
            all_pred.append(norm_pred)
            
            # Keep track of individual samples for debugging
            logs.append({
                "id": i,
                "ref": norm_gt,
                "pred": norm_pred,
                "wer": jiwer.wer(norm_gt, norm_pred)
            })

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
            
    # 3. Comprehensive Metric Calculation
    if not all_gt:
        print("No valid data processed.")
        return

    wer, cer = calculate_metrics(all_gt, all_pred)

    # 4. Reporting
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