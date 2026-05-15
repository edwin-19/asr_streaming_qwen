from fastapi import APIRouter, Form, Depends, Request, UploadFile, File, HTTPException
import io
import soundfile as sf
import numpy as np
import librosa

from schema.transcription import TranscriptionResponse, WordSegment

transcribe_route = APIRouter(prefix="/v1/audio")

@transcribe_route.post("/predict", response_model=TranscriptionResponse)
async def transcribe(request: Request, file: UploadFile = File(...)):
    # Access the shared model from the app state
    model = request.app.state.asr_model
    audio_bytes = await file.read()
    data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    
    if samplerate != 16000:
        data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
    
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
        
    results = model.transcribe(audio=(data, 16000), return_time_stamps=True)
    res_obj = results[0]
    return {
        "language": res_obj.language,
        "text": res_obj.text,
        "words": [
            {
                "word": item.text,
                "start": item.start_time,
                "end": item.end_time
            } for item in res_obj.time_stamps.items # Access the .items list inside ForcedAlignResult
        ]
    }

@transcribe_route.post('/pred_timestamp')
async def get_timestamp(request: Request, text: str, language:str, file: UploadFile = File(...)):
    model = request.app.state.asr_model.forced_aligner
    audio_bytes = await file.read()
    data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    
    if samplerate != 16000:
        data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
    
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    results = model.align(
        audio=(data, 16000), text=text, language=language
    )[0]
    
    return results