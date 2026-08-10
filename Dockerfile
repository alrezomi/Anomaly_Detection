FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/alrezomi/Anomaly_Detection"
LABEL org.opencontainers.image.description="DINOv2 vision and GMR time-series anomaly detection"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    HF_HOME=/cache/huggingface

WORKDIR /workspace/anomaly_detection

ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"

COPY . .

ENTRYPOINT ["python", "launch_pipeline.py"]
CMD ["both"]
