---
name: pipeline-down
description: Tear down the local ImageGenie pipeline Docker stack (download/convert/normalize/render workers, Postgres, Pub/Sub emulator) and remove its volume. Use when done smoke-testing the local pipeline, or when you want a clean slate before re-running it. Complements the pipeline-up skill.
---

# Tear down the local pipeline

```
docker compose -f server/docker-compose.yml down -v
```

(`make compose-down` is the shortcut.)

This stops and removes every service container, the network, **and** the
`storage` volume. Removing the volume also drops the Postgres data — which is
exactly what you want for a clean next run, since the pipeline is idempotent and
would otherwise skip any already-processed models on the next `pipeline-up`.

## Notes

- **No cloud resources are involved** — this is purely local Docker. (Unrelated to
  the always-on Cloud SQL instance from milestone 3, which is torn down separately
  via `terraform -chdir=infra destroy`.)
- `render_captures/` on the host **persists** (it's a local analysis output,
  gitignored) — delete it by hand if you want it gone.
- Verify nothing is left:
  ```
  docker compose -f server/docker-compose.yml ps
  ```
