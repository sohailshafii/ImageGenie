# Training image for the Vertex AI custom job (M6 chunk G, server.md#training-gpu).
#
# Separate from server/Dockerfile rather than a variant of it. That image carries
# the mesh stack (OSMesa, X11, the PyOpenGL fork) and the built SPA, none of which
# training touches, and it installs the CPU torch wheels — which would silently
# produce a GPU-less job on a GPU we are paying for. The two images share only the
# `app` package, copied in below.
#
# Built with Cloud Build, not locally: the base is a multi-GB CUDA image and this
# host is arm64, so a local `--platform linux/amd64` build runs under emulation.
# See `make train-image`.

# torch + torchvision come from the base, pinned by PyTorch against a matching
# CUDA runtime — hence no torch line in requirements-train.txt. `-runtime` (not
# `-devel`) because nothing here compiles CUDA kernels.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /srv

# Fail fast and unbuffered: a Vertex job's only console is the log stream, so
# output has to arrive as it happens rather than at exit.
ENV PYTHONUNBUFFERED=1

COPY ml/requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# Both packages, flat on the path — `ml/train.py` imports `app.db`/`app.models`
# (bookkeeping) and `taxonomy` (the class roster), the same two-way split the
# `train` and `cleanup-raw` make targets set up with PYTHONPATH locally.
COPY server/app ./app
COPY ml/*.py ./

# Training reads renders from the processed bucket and writes weights back to it.
ENV IMAGEGENIE_STORAGE_BACKEND=gcs

ENTRYPOINT ["python", "train.py"]
