# FINAL STAGE
FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-devel 
WORKDIR /app

# Set PYTHONPATH
ENV PYTHONPATH=/app:/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . /app/

# Install Python requirements with the bypass flag
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt
RUN pip install --no-cache-dir --break-system-packages qwen-asr[vllm]

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]