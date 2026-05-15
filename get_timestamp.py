import typer
from datasets import load_dataset
from qwen_asr import Qwen3ForcedAligner
import torch
from tqdm import tqdm

app = typer.Typer()

@app.command()
def main(
    dataset_path:str=typer.Option("./data/seame_dev_sge"),
    model_path:str=typer.Option("./models/Qwen3-ForcedAligner-0.6B"),
    device:str=typer.Option("cuda")
):
    dataset = load_dataset(dataset_path)
    model = Qwen3ForcedAligner.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device)

    alignment_data = []
    for d in dataset['test']:
        results = model.align(
            audio=(d['context']['array'], d['context']['sampling_rate']), 
            text=d['answer'], language=['Chinese']
        )
        for item in results[0].items:
            alignment_data.append({
                "word": item.text,
                "start": round(item.start_time, 3),
                "end": round(item.end_time, 3),
                "duration": round(item.end_time - item.start_time, 3)
            })    
    
if __name__ == '__main__':
    app()