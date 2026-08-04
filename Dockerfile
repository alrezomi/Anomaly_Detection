FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/alrezomi/Anomaly_Detection"
LABEL org.opencontainers.image.description="DINOv2 vision and GMR time-series anomaly detection"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    HF_HOME=/cache/huggingface

WORKDIR /workspace/AD/Script_VS

ARG TORCH_VERSION=2.11.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir \
    "torch==${TORCH_VERSION}" \
    --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "launch_pipeline.py"]
CMD ["both"]
