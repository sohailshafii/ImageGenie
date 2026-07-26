---
name: app-up
description: Stand up the ImageGenie app stack locally (throwaway Postgres → migrations → seed → FastAPI → Vite) so a frontend change can be checked in a real browser, then tear it back down. Use before browser-verifying any web/ or API change. Distinct from pipeline-up, which runs the worker pipeline instead.
---

# Bring the app up for a browser check

Everything runs from the repo root unless stated. Takes ~30s to a usable page.

**This is not `pipeline-up`.** That skill runs the preprocessing workers via
`server/docker-compose.yml`. This one runs the *app* — API + SPA + a database —
and deliberately does **not** use compose, because compose's Postgres publishes
no host port, so `alembic` and `uvicorn` on the host can't reach it.

## 1. Throwaway Postgres

```
docker rm -f imagegenie-verify 2>/dev/null
docker run -d --name imagegenie-verify -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 postgres:16-alpine
sleep 6
docker exec -i imagegenie-verify psql -U postgres \
  -c "CREATE USER imagegenie WITH PASSWORD 'imagegenie' SUPERUSER;" \
  -c "CREATE DATABASE imagegenie OWNER imagegenie;"
```

The role/database must be created by hand: `POSTGRES_PASSWORD` only provisions
the `postgres` superuser, but `Settings.database_url` defaults to
`postgresql+psycopg://imagegenie:imagegenie@localhost:5432/imagegenie`.

## 2. Schema

```
cd server && ../.venv/bin/alembic upgrade head && cd ..
```

(`make migrate` is the same thing.) Expect one line per revision.

## 3. Seed

Write a throwaway script in the scratchpad and run it with `PYTHONPATH=server`:

```
PYTHONPATH=server .venv/bin/python <scratchpad>/seed_x.py
```

Always create the admin — signup is invite-gated, so a fresh DB has no way in:

```python
from app.create_admin import create_admin
create_admin("admin@imagegenie.dev", "genie-admin")
```

Add a viewer directly (`User(..., role=UserRole.user, verified=True)`) whenever
the change has a role-dependent surface. Seed **blobs** through
`build_storage(get_settings())` so keys match `artifact_keys`; storage is
`LocalStorage` under `data/storage/`, relative to the **cwd of the API process**.

Gotchas that have bitten:
- A raw-SQL model insert needs an explicit `download_status` — its default is
  ORM-side, not a DB default. Prefer the ORM.
- Seed the *interesting* states, not just the happy one: missing artifacts, nulls,
  every status, out-of-order timestamps. That is what the browser check is for.

## 4. Servers

Both in the background, from the repo root (so `data/storage` resolves):

```
PYTHONPATH=server .venv/bin/python -m uvicorn app.api:app --port 8000
cd web && npm run dev        # → http://localhost:5173
```

Vite proxies `/api` **and** `/artifacts` to :8000 (see `vite.config.ts`). Browse
to **5173**, never 8000 — the SPA is not served by uvicorn in dev.

Wait for readiness rather than sleeping blindly:

```
for i in 1 2 3 4 5 6 7 8; do
  /usr/bin/curl -sf http://localhost:8000/healthz >/dev/null && echo "API up" && break
  sleep 2
done
```

Use the absolute `/usr/bin/curl` — a bare `curl` has failed to resolve inside
`for` loops in this environment.

## 5. Check the endpoints before the browser

Cheaper than clicking, and localises failures to server vs. client:

```
/usr/bin/curl -s -c jar.txt -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@imagegenie.dev","password":"genie-admin"}' \
  -o /dev/null -w "login %{http_code}\n"
/usr/bin/curl -s -b jar.txt -D - -o /dev/null http://localhost:8000/<path> | grep -iE "^HTTP"
```

Then hand off to the **claude-in-chrome** skill: log in at
`http://localhost:5173/`, exercise the change, and check
`read_console_messages` for errors. Verify each role separately when a gate is
involved — sign out and back in as the viewer.

**Logging in is the flaky step.** Chrome's autofill dropdown opens over the
password field and swallows `type` keystrokes, and `form_input` refs go stale
when the form re-renders — both fail *silently*, leaving you staring at a login
page wondering why. The reliable sequence:

1. `key` **Escape** to dismiss any autofill popup.
2. `find` the fields again — never reuse refs from before a re-render.
3. `form_input` each field, then click the submit button by ref.
4. **Confirm it worked from the server side**, not the screenshot:
   `grep "auth/login" <scratchpad>/api.log` should show a `200 OK`. Repeated
   `GET /auth/me 401` and no `POST /auth/login` means the keystrokes never landed.

To reach a state the seed data can't produce (an empty filter, a missing
artifact), mutate the DB directly with `docker exec -i imagegenie-verify psql -U
imagegenie -d imagegenie -c "..."` and reload — faster than reseeding, and the
container is thrown away anyway. `training_metric` has an FK to `training_run`,
so delete metrics first.

## 6. Tear down

```
docker rm -f imagegenie-verify
rm -rf data/storage
pkill -f "uvicorn app.api"; pkill -f vite
```

Also delete anything the run dropped in `~/Downloads`. The backgrounded servers
report exit code 143/144 when killed — that is the teardown working, not a
failure.
