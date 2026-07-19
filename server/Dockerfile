# Worker/seed image for the pipeline (milestones 2 + 4). One image runs every
# stage; the compose service / Cloud Run job overrides CMD to pick the worker.
FROM python:3.12-slim

WORKDIR /srv

# System libs for the render stage. pyrender renders headless (no display) through
# OSMesa — a pure-software OpenGL rasterizer that needs no GPU, X display, or EGL
# device, which makes it the deterministic choice in a slim container (EGL's
# headless device path is fragile here). libGL/libglib are runtime deps; `git` is
# only needed to pip-install the PyOpenGL fork below. The libX11/libXrender/... libs
# are NOT for rendering: `import pyrender` unconditionally imports its pyglet-based
# Viewer, which links the X11 bindings at import time, so they must be present even
# though we only ever use the offscreen renderer and never open a window.
# PYOPENGL_PLATFORM=osmesa tells PyOpenGL to render through OSMesa.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libosmesa6 \
        libgl1 \
        libglib2.0-0 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libxi6 \
    && rm -rf /var/lib/apt/lists/*
ENV PYOPENGL_PLATFORM=osmesa

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Default: run the download worker (consumes the queue forever). Preprocessing
# stages override this, e.g. `python -m app.workers.render`.
CMD ["python", "-m", "app.workers.download"]
