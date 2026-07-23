# Worker + API image. One image runs every pipeline stage (the compose service /
# Cloud Run job overrides CMD to pick a worker) AND the backend-for-frontend API
# (uvicorn app.api:root_app), which serves the built SPA — so the build compiles
# the frontend too. Built from the repo root so both server/ and web/ are in the
# context (see `make deploy-image` and docker-compose.yml).

# ── Stage 1: build the single-page app (web/dist) ───────────────────────────
# The API serves this build so the SPA and API share one origin
# (server.md#serving-the-spa). Compiling it here keeps the image self-contained;
# with layer caching the `npm ci` step is skipped unless the lockfile changes, so
# iterating on the workers doesn't pay for it.
FROM node:22-bookworm-slim AS web-build
WORKDIR /web
# Deps first, against the lockfile, so this layer caches independently of source.
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build  # tsc -b && vite build -> /web/dist

# ── Stage 2: worker + API runtime ───────────────────────────────────────────
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

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/app ./app

# The built SPA the API serves. IMAGEGENIE_SPA_DIR points the API (root_app) at
# it; the workers never read it, so it is inert for them.
COPY --from=web-build /web/dist ./web_dist
ENV IMAGEGENIE_SPA_DIR=/srv/web_dist

# Default: run the download worker (consumes the queue forever). Preprocessing
# stages override this, e.g. `python -m app.workers.render`; the API service runs
# `uvicorn app.api:root_app`.
CMD ["python", "-m", "app.workers.download"]
