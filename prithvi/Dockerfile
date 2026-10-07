# Prithvi-EO-2.0 (Sen4Map crops) service. Self-contained: the exact package set
# of the image the checkpoint was verified in (requirements.lock = its pip freeze).
# Build context is the repo root: docker compose --profile prithvi build prithvi
FROM python:3.11.15-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libexpat1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/mpl \
    AGRISENSE_MAX_PX=0 \
    HF_HUB_OFFLINE=1 \
    PRITHVI_CKPT=/models/prithvi_sen4map_crops.ckpt

WORKDIR /app
RUN pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
COPY prithvi/requirements.lock .
RUN pip install -r requirements.lock

COPY satellite.py prithvi/service.py prithvi/validate_worldcover.py /app/
EXPOSE 8601
CMD ["python3", "service.py"]
