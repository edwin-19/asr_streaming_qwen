import typer
from pathlib import Path
import json
from tqdm import tqdm
import os
import re
from lingua import Language, LanguageDetectorBuilder

app = typer.Typer()

def norm_spacing(text):
    # 1. Join Chinese characters (removes spaces between them)
    # Matches a Chinese character, followed by spaces, followed by another Chinese character
    # We use a lambda to ensure we don't miss overlapping matches like "A B C"
    pattern = re.compile(r'([\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])')
    text = pattern.sub(r'\1', text)
    
    # 2. Clean up English/Chinese boundaries (Optional but recommended)
    # Ensures "东西BUT" becomes "东西 BUT" and "BUT不可以" becomes "BUT 不可以"
    text = re.sub(r'([\u4e00-\u9fff])([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])([\u4e00-\u9fff])', r'\1 \2', text)
    
    # 3. Fix double spaces and casing
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def read_jsonl(json_path):
    with open(json_path, 'r') as jsonf:
        return [json.loads(data) for data in jsonf.readlines()]


def write_jsonl(json_path, data):
    fdata = [json.dumps(d, ensure_ascii=False) + '\n' for d in data]
    with open(json_path, 'w') as jsonl:
        jsonl.writelines(fdata)

@app.command()
def main(
    data_path:str=typer.Option("data/seame.json")
):
    metadata = read_jsonl(data_path)
    all_data = []
    languages = [Language.ENGLISH, Language.CHINESE]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    
    for meta in tqdm(metadata):
        audio_fpath = os.path.join('data/new_seame', os.path.basename(meta['audio_filepath']))
        if os.path.exists(audio_fpath):
            text = norm_spacing(meta['text']).lower()
            language = detector.detect_language_of(text)
            if language == Language.CHINESE:
                lang = 'Chinese'
            else:
                lang = 'English'
                
            all_data.append({
                'audio': audio_fpath,
                'text': f'{lang}<asr_text>{text}'
            })
        
    write_jsonl("data/seame_train_qwen_asr.json", all_data)
        
if __name__ == "__main__":
    app()