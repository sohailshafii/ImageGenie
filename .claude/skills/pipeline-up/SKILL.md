---
name: pipeline-up
description: Bring up the local ImageGenie preprocessing pipeline (download → convert → normalize → render) in Docker and smoke-test it end to end — seed a few models and eyeball the multi-view renders. Use when asked to run/start the local pipeline, reproduce the compose stack, or verify the render stage actually produces images (the unit tests mock the GL call, so only this exercises real headless rendering).
---

# Bring up the local pipeline (smoke test)

The stack (`server/docker-compose.yml`) is Postgres + a Pub/Sub emulator + one
worker per stage: `download → convert → normalize → render`, chained over per-stage
Pub/Sub topics. It runs the same worker code that runs on Cloud Run. **Local Docker
only — no cloud cost.**

## 1. Build + start

```
docker compose -f server/docker-compose.yml up -d --build
```

First build is a few minutes: the image installs OSMesa + X11 libs and a PyOpenGL
fork so the render stage can rasterize **headless** (see `server/Dockerfile`).
`make compose-up` is the shortcut.

## 2. Verify every stage is consuming

```
docker compose -f server/docker-compose.yml ps
docker compose -f server/docker-compose.yml logs render   # expect "consuming render-worker"
```

All of `worker` (download), `convert`, `normalize`, `render` should log
`consuming …`. If `render` crash-loops, check its logs for OpenGL/OSMesa errors.

## 3. Seed jobs — from a FRESH-code container

```
docker compose -f server/docker-compose.yml run --rm worker python -m app.seed --count 5
```

Seeds N download jobs that flow through all four stages.

> ⚠️ **Do not seed via the `seed` compose service.** It sits behind a compose
> profile, is not rebuilt by `up`, and its cached image can be stale (missing newer
> `app/` code). Run the seed (or any ad-hoc publish) from a freshly-built stage
> container — `worker`, `convert`, `normalize`, or `render` — instead. `make
> compose-seed` uses the `seed` service, so add `--build` or use the above after
> code changes.

## 4. Watch it work

```
# per-stage artifact counts (download writes `model`; the rest write `artifact`)
docker compose -f server/docker-compose.yml exec -T postgres \
  psql -U imagegenie -tA -c "select stage, status, count(*) from artifact group by 1,2 order by 1"
docker compose -f server/docker-compose.yml logs -f render
```

Expect `converted`, `normalized`, `rendered` each reaching N, and 12 `view_NN.png`
per model under `/data/storage/processed/renders/<uid>/`.

## 5. Eyeball the renders

```
.venv/bin/python capture_renders.py [uid]     # defaults to the first model
```

Writes a 12-view contact sheet to `render_captures/<uid>_contact_sheet.png`
(gitignored). Read that PNG to inspect shape/shading.

## Gotchas

- **Idempotency skips re-seeds.** Re-seeding the same uids skips at *every* stage
  (including render, if its full view set exists). For a clean re-run, tear the
  volume down first (see the `pipeline-down` skill — `down -v` also clears
  Postgres). To re-render *one* model without a full reset: delete its `rendered`
  artifact row and its `processed/renders/<uid>/` blobs, then publish a render job
  from a fresh-code container:
  ```
  docker compose -f server/docker-compose.yml exec -T render python -c \
    "from app.config import get_settings; from app.queue import publish_next; \
     publish_next(get_settings().render_topic, '<uid>')"
  ```
- **Render only works in Docker.** pyrender's OSMesa path needs the container's
  system libs; it will not render on the host Mac. The pytest suite mocks the GL
  call, so compose is the only place the real renderer runs.
