# Rostee image: API + frontend with all three models baked in, so the
# container starts without downloading anything.

# slim = Debian with only what Python needs (~130 MB instead of ~1 GB).
# 3.12 matches the .venv the code is developed and tested on.
FROM python:3.12-slim

# PYTHONUNBUFFERED: print logs right away, so `docker compose logs` is live.
# PYTHONDONTWRITEBYTECODE: no .pyc files; the code folder is read-only anyway.
# PIP_NO_CACHE_DIR: pip's download cache would only make the image bigger.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Every step is cached and only re-run when its inputs change. Requirements
# change least, so they go first: editing code won't reinstall every package.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run as a normal user, not root: if someone breaks into the app they can't
# take over the container. It owns only the two folders it writes to.
RUN useradd --create-home --uid 1000 rostee \
 && mkdir -p /opt/models /app/data \
 && chown rostee:rostee /opt/models /app/data
USER rostee

# Models live outside /app/data on purpose: the data volume is mounted over
# /app/data at run time and would hide anything the build put there.
ENV MODELS_CACHE_DIR=/opt/models

# Copy only what the model download imports, so editing api.py or the
# pipeline doesn't re-download ~150 MB of models on the next build.
COPY app/__init__.py app/config.py app/schemas.py app/
COPY app/core/__init__.py app/core/embedder.py app/core/reranker.py app/core/

# Model names come from config.py, the one place they're defined. Settings
# insists on a Gemini key and a password that the download never uses, so
# dummies for this one command are enough; the real ones arrive at run time.
RUN GEMINI_API_KEY=build-only APP_PASSWORD=build-only python -c "\
from app.config import get_settings; \
from app.core.embedder import Embedder; \
from app.core.reranker import Reranker; \
s = get_settings(); \
Embedder(dense_model=s.dense_model, sparse_model=s.sparse_model, dense_dim=s.dense_dim, \
         cache_dir=s.models_cache_dir, threads=s.model_threads, batch_size=s.model_batch_size); \
Reranker(model=s.reranker_model, cache_dir=s.models_cache_dir, \
         threads=s.model_threads, batch_size=s.model_batch_size)"

# From here on, never contact Hugging Face: a missing model fails loudly at
# startup instead of quietly downloading on a client's server.
ENV HF_HUB_OFFLINE=1

# Linux's memory allocator keeps freed memory for reuse instead of giving it
# back, so after one big upload the app stays big. These make it return memory
# sooner: a 330-page upload ended at 385 MB instead of ~450 MB. Set here, late,
# because it only matters at run time, and an ENV near the top would make
# every build reinstall all packages.
ENV MALLOC_ARENA_MAX=2 \
    MALLOC_TRIM_THRESHOLD_=65536

# The rest of the code. COPY makes files owned by root, so the app can't
# change its own code even though it runs as the rostee user.
COPY app/ app/
COPY frontend/ frontend/

EXPOSE 8000

# 0.0.0.0, not 127.0.0.1: inside a container 127.0.0.1 only accepts
# connections from the container itself, so your browser couldn't reach it.
# Hosts like Render pick the port and pass it in $PORT; locally it's 8000.
# `exec` makes uvicorn the main process, so it gets Docker's stop signal
# and shuts down cleanly instead of being killed.
CMD ["sh", "-c", "exec uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
