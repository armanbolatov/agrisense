# AgriSense image.
#
# Default build is GPU (CUDA 12.8 wheels, which support the RTX 5090 / Blackwell
# sm_120). Installing torch + torchvision together from the SAME index pins an
# ABI-matched pair — that is what fixes the original
# "operator torchvision::nms does not exist" error that broke SAM.
#
# CPU build:  docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .

FROM python:3.11-slim

# Small runtime libs a couple of wheels load (GDAL/PROJ/GEOS are bundled in the
# rasterio/pyproj/shapely wheels; libgomp is needed by torch).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /app

# torch + torchvision (matched pair from one index). cu128 = GPU by default.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        --index-url ${TORCH_INDEX}

# Everything else.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code (changes most often → last, for best layer-cache reuse).
COPY . .

EXPOSE 8000
CMD ["python", "server.py"]
