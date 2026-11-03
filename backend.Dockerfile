ARG BASE_IMAGE=pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime
ARG MODEL_SIZE=base_plus

FROM ${BASE_IMAGE} AS dependencies

# Gunicorn environment variables
ENV GUNICORN_WORKERS=1
ENV GUNICORN_THREADS=2
ENV GUNICORN_PORT=5000

# SAM 2 environment variables
ENV APP_ROOT=/opt/sam2
ENV PYTHONUNBUFFERED=1
ENV SAM2_BUILD_CUDA=0

# Install system requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    pkg-config \
    build-essential \
    libffi-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Reuse the image's PyTorch while installing application dependencies in a venv.
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR ${APP_ROOT}
COPY setup.py pyproject.toml MANIFEST.in README.md LICENSE LICENSE_cctorch ./
COPY sam2 ./sam2
COPY training ./training

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir --no-build-isolation ".[interactive-demo]"

FROM dependencies AS runtime
ARG MODEL_SIZE
ENV MODEL_SIZE=${MODEL_SIZE}

# Copy backend server files
COPY demo/backend/server ${APP_ROOT}/server

# Download SAM 2.1 checkpoints
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_tiny.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_small.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_base_plus.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_large.pt

WORKDIR ${APP_ROOT}/server

# https://pythonspeed.com/articles/gunicorn-in-docker/
CMD gunicorn --worker-tmp-dir /dev/shm \
    --worker-class gthread app:app \
    --log-level info \
    --access-logfile /dev/stdout \
    --log-file /dev/stderr \
    --workers ${GUNICORN_WORKERS} \
    --threads ${GUNICORN_THREADS} \
    --bind 0.0.0.0:${GUNICORN_PORT} \
    --timeout 60
