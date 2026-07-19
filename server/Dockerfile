# Worker/seed image for the pipeline (milestones 2 + 4). One image runs every
# stage; the compose service / Cloud Run job overrides CMD to pick the worker.
FROM python:3.12-slim

WORKDIR /srv

# System libs for the render stage: OSMesa gives pyrender a headless (no display)
# OpenGL context, and libGL/libglib are its runtime deps. PYOPENGL_PLATFORM=osmesa
# tells PyOpenGL to use it instead of a windowed GL driver.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ENV PYOPENGL_PLATFORM=osmesa

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Default: run the download worker (consumes the queue forever). Preprocessing
# stages override this, e.g. `python -m app.workers.render`.
CMD ["python", "-m", "app.workers.download"]
