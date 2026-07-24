# Server / Infra — ImageGenie

Backend of the pipeline: ingestion, the queue + workers preprocessing chain, the API layer,
object storage, and the metadata database. See [../CLAUDE.md](../CLAUDE.md) for the project hub.

## Scope

- **Download workers** — pull models from the model store API (FR-1). Polite rate limiting,
  no scraping, resumable.
- **Conversion workers** — STL/OBJ/GLB/FBX → a common internal format.
- **Normalize workers** — center, rescale to a unit bounding box, validate mesh integrity.
- **Render workers** — produce multi-view images (v1) or point clouds (stretch).
- **API layer** — FastAPI. Enqueues jobs, serves labels/metadata/results to the frontend,
  handles auth (see [web.md](../web/web.md#auth--roles)).
- **Storage** — object storage for raw + processed artifacts; metadata DB for everything queryable.

## Ingestion Source

Models come from **Objaverse** (~800k Sketchfab-sourced 3D objects), pulled via its official Python
package, which downloads from a hosted mirror rather than scraping Sketchfab directly. Chosen
because it (a) permits bulk download for personal ML use without ToS friction, (b) exposes the
tags/categories/titles the [weak-label policy](../ml/ml.md#weak-label-policy) needs, and (c) sidesteps
rate-limit/scraping concerns. Download workers (FR-1) call this package with polite rate limiting and
resumable, idempotent fetches — see [Request Resilience](#request-resilience) for the backoff/retry
policy.

> **TODO:** Make download a separately **controlled** stage — deliberately triggered/gated (run in
> batches we kick off), not fired automatically end-to-end. Data can also enter the pipeline via
> **manual upload** (admins uploading models directly, FR-9 / [web.md](../web/web.md#data-upload)) as
> an alternative to the Objaverse crawl.

## Cloud Platform (GCP)

The project runs on **Google Cloud Platform** as its single ecosystem (resolves the cloud-provider
[open decision](../CLAUDE.md#open-decisions)). GCP was chosen over AWS because the workload is
**bursty preprocessing on a hard ~$100 budget**: GCP's serverless compute scales to zero and its
managed queue costs nothing at idle, so we pay only while jobs actually run. Keeping every service in
one ecosystem also avoids cross-cloud egress and IAM sprawl.

Each pipeline component maps to a GCP service. Rationale is recorded per component below as we settle
it — walking the list one at a time:

| Component | Role | GCP service (candidate) | Decision |
|-----------|------|-------------------------|----------|
| Compute | runs the workers | Cloud Run | **settled** — see [Compute](#compute) |
| Queue | passes jobs between stages | Pub/Sub | **settled** — see [Queue](#queue) |
| Object storage | raw + processed blobs | Cloud Storage (GCS) | **settled** — see [Object storage](#object-storage) |
| Database | metadata / labels / runs | Cloud SQL (Postgres) | **settled** — see [Database](#database) |
| Training GPU | trains the CNN (milestone 6) | Vertex AI (spot) | **settled** — see [Training GPU](#training-gpu) |

### Compute

Workers run on **Cloud Run**, using both of its execution flavors — matched to the shape of each
stage rather than forcing one pattern:

- **Download stage (FR-1) → Cloud Run *job*.** Fetching 20–50k models from Objaverse is a single,
  sustained, politely-rate-limited crawl — batch-shaped, not per-message. A run-to-completion job
  pulls a list of model ids and grinds through it, then exits, without thrashing request-driven
  autoscaling.
- **Preprocessing stages (convert → normalize → render) → Cloud Run *services* behind Pub/Sub push.**
  These are bursty, short (seconds–minutes per model), and embarrassingly parallel. Each queued job
  is delivered as one HTTP request; the service autoscales instances with queue pressure and
  **scales to zero** when idle — the core of the pay-only-while-running cost story.

The same container image runs in either flavor, so the local Docker skeleton (milestone 2) exercises
identical code against the Pub/Sub emulator. The 60-min Cloud Run *service* request timeout is ample
for per-model preprocessing. All four stages run the **one** image and push receiver
(`app.web:app`); `IMAGEGENIE_STAGE` selects which stage's `process` handles the message, so a single
service definition (fanned out with Terraform `for_each` in `infra/preprocessing.tf`) serves each
stage — each with its own topic/subscription/DLQ and its `STAGE` env, download included.

**One model per instance.** Each service runs at `max_instance_request_concurrency = 1` with 2Gi RAM.
A pilot ingestion showed why: objaverse's downloader and trimesh/pyrender are memory-heavy and not
safe to run many-to-an-instance — concurrent big meshes OOM'd the 512Mi default, and the OOM-kills
truncated objaverse's on-disk cache mid-write (corrupt files on retry). One request per instance fixes
both and bounds each instance to ~1 DB connection; throughput scales by adding instances, not
in-instance concurrency. Cloud SQL's `max_connections` is raised to 100 to cover the fan-out.

### Queue

Stages are connected by **Pub/Sub**, resolving the queue-technology
[open decision](../CLAUDE.md#open-decisions) in favour of the managed option over self-hosted
Redis+Celery (which would mean paying to run Redis 24/7 plus rebuilding retry/DLQ machinery — a
standing idle cost that fights NFR-1).

**What Pub/Sub is.** A fully-managed message broker built on the *publish/subscribe* pattern. A
**publisher** sends a message to a named **topic** without knowing who consumes it; a **subscriber**
attaches a **subscription** to that topic to receive copies. It decouples producers from consumers and
durably **buffers** messages, so a burst of thousands of jobs is held safely until workers catch up.
Google runs all storage/delivery/scaling; we just create topics and subscriptions.

**Topology** — one topic + subscription per stage boundary; each subscription has its own dead-letter
topic:

```
download job ──▶ [convert topic] ──▶ convert svc
                                      └▶ [normalize topic] ──▶ normalize svc
                                                              └▶ [render topic] ──▶ render svc
each subscription → its own dead-letter topic (error recorded in the DB)
```

Workers use **push** subscriptions (Pub/Sub delivers each message as an HTTP request to the Cloud Run
service), which pairs with scale-to-zero. Pull subscriptions were rejected: they need an
always-running puller, undercutting scale-to-zero.

> **Skeleton exception (milestone 2).** The local download worker uses a **pull** subscription. The
> objection to pull is purely scale-to-zero cost, which doesn't apply locally, and the download stage
> is a batch consumer anyway (a Cloud Run *job* in prod, not a push service). Prod preprocessing stages
> keep push.

**At-least-once delivery.** Pub/Sub guarantees every message is delivered *one or more* times — never
zero, occasionally twice. A worker must **acknowledge ("ack")** a message when done; if it fails to
ack within the deadline (crash, timeout), Pub/Sub **redelivers**. A model can therefore be processed
more than once — which is *precisely why every worker must be idempotent* (NFR-2). (At-most-once loses
messages; exactly-once is costly and brittle — at-least-once + idempotent handlers is the standard
robust combo.)

**Dead-letter (DLQ).** Some messages fail every attempt — a corrupt mesh, an unconvertible file
("poison" messages). Without a backstop they'd redeliver forever, blocking the stage. A **dead-letter
topic** is a separate quarantine topic where Pub/Sub *automatically* routes a message once it has
exceeded a configured max number of delivery attempts. The failed model lands there and its error is
recorded in the DB — never silently dropped, never infinitely retried.

### Object storage

Heavy binary artifacts live in **Cloud Storage (GCS)** — object storage that holds arbitrary blobs
("objects") in **buckets**, each retrieved by a key/path. It is not a filesystem and not a database:
the [metadata DB](#database) stores only the object keys, never the blobs themselves.

**Layout — two same-region buckets:**

- `imagegenie-raw` — source meshes: `raw/<uid>.<ext>`.
- `imagegenie-processed` — converted / normalized / rendered outputs, separated by prefix.

**The source mesh's format is carried by its key**, not assumed. Ingestion only ever writes GLB
(that is what Objaverse serves), but an [admin upload](../web/web.md#data-upload) may be STL or OBJ,
so `convert` reads `model.raw_key` and derives the trimesh `file_type` from its extension
(`app/artifact_keys.py`, `RAW_SUFFIX_TO_FILE_TYPE`) rather than hardcoding one. A row with no
`raw_key` falls back to `.glb`, so rows written before uploads existed behave exactly as before.

> **FBX is not supported**, despite FR-2 and FR-9 listing it. trimesh has no FBX loader — the format
> is proprietary and undocumented — so it is rejected at upload with a clear error rather than
> dead-lettering deep in the convert stage. Nothing in the corpus is affected: all 12,029 ingested
> models are GLB, so the claim was never exercised. Adding it later means an `assimp` package in the
> worker image, invoked as a subprocess rather than through `pyassimp` (whose bindings must match the
> installed library version, and which would put a memory-unsafe parse in-process on the one path
> that accepts untrusted files), plus one entry in `RAW_SUFFIX_TO_FILE_TYPE`. No other stage assumes
> a format.

Two buckets (not one) because raw and processed have different lifecycles: raw is deleted or
cold-stored independently once a model is preprocessed or excluded, while processed stays hot for
training. Both buckets sit in a **single region colocated with Cloud Run + Cloud SQL** (e.g.
`us-central1`) so same-region reads are free — directly serving NFR-5 (bring code to data, avoid
egress). Multi-region would cost more for no benefit here.

**Storage classes — Standard vs. Nearline.** GCS storage classes trade storage price against access
price plus a minimum-duration commitment:

- **Standard** — highest per-GB storage price, but **no retrieval fee** and **no minimum storage
  duration**. For "hot" data read frequently. → **processed** renders, which training reads every
  epoch, stay Standard.
- **Nearline** — roughly **half** the per-GB storage price, but charges a **per-GB retrieval fee** on
  every read and imposes a **30-day minimum storage duration** (delete sooner and you're still billed
  for 30 days). For data touched less than monthly. → **raw** meshes, rarely re-read once
  preprocessed, transition to Nearline.

(Coldline / Archive go cheaper on storage with steeper retrieval fees and 90-/365-day minimums —
overkill here.) The trap avoided: putting training data in Nearline would rack up per-epoch retrieval
fees and erase the savings, so only raw is cold-stored.

**Lifecycle rule** on `imagegenie-raw`: transition to Nearline once preprocessed and delete models
excluded from the dataset outright (cost guardrail) — keeps the ~$5–15/mo storage line in check.

**Client abstraction.** Workers reach storage through a thin `Storage` protocol
(`server/app/storage.py`) addressed by **key** (e.g. `raw/<uid>.glb`), never touching buckets/paths
directly. The milestone-2 skeleton uses `LocalStorage` over a local directory (one root, key prefixes
distinguish raw/processed); in cloud `RoutedGcsStorage` routes by key prefix — `raw/*` → the raw
bucket, everything else (`processed/*`) → the processed bucket — so the two-bucket split is invisible
to worker code, which is unchanged between local and GCS.

### Training GPU

The milestone-6 multi-view CNN (a pretrained ResNet fine-tuned on renders) trains as a **Vertex AI
custom training job on a spot GPU**, not a hand-managed VM. Vertex provisions the GPU **only for the
job's duration and releases it automatically** on completion — the same pay-only-while-running
principle used for compute and the queue, and the structural guard against this project's biggest
budget risk: a forgotten, idle GPU VM. Spot pricing keeps the training line at $5–20 total; the job
container is the same PyTorch image, checkpointing to GCS so a spot **preemption** simply resumes
(see the [ML coding standard](../ml/ml.md#coding-standards-ml)).

**GPU: default T4** — the cheapest widely-available spot GPU, ample for a small multi-view CNN. Step
up to **L4** only for faster/newer silicon if spot availability is good. A plain Compute Engine spot
VM was rejected: cheaper per-hour but requires manual teardown, reintroducing the idle-GPU risk.

## Queue + Workers

- Embarrassingly parallel; scale by adding workers (NFR-3).
- **Idempotency (NFR-2)** — every worker checks whether its output already exists (by content
  hash / model id + stage) and skips if so. Reruns must be safe. This is the single most
  important backend invariant — a crashed batch must be re-runnable without duplicating work.
- **At-least-once delivery** — assume messages can be redelivered; design handlers to tolerate it.
- **Dead-letter handling** — models that fail a stage N times go to a dead-letter queue with the
  error recorded in the DB, not silently dropped.
- Queue technology is **Pub/Sub** (managed) — see [Queue](#queue) for topology and delivery semantics.
- **Implementation.** Each worker is a module with a single `process(job)` entrypoint under
  `server/app/workers/`, and a `main()` that calls the shared `run_stage()` (bootstrap DB +
  subscription, then consume). Each stage hands the model to the next by publishing `{"uid"}` to the
  next topic (`publish_next`), so the chain is `download → convert → normalize → render`:
  - `download.py` — fetch the mesh, upsert the `model` row (`INSERT ... ON CONFLICT (uid)`).
  - `convert.py` — flatten the raw GLB to a single mesh and re-export it as canonical **PLY**
    (`processed/converted/<uid>.ply`).
  - `normalize.py` — center on the bounding-box center and scale the largest extent to 1
    (`processed/normalized/<uid>.ply`).
  - `render.py` — render 12 views (224²) around the object with **trimesh + pyrender** offscreen
    (OSMesa in the container), writing `processed/renders/<uid>/view_NN.png` (terminal stage).

  The preprocessing stages share `workers/mesh.py` (load/concatenate/export) and `workers/artifacts.py`
  (the `(model_uid, stage)` idempotency gate + upsert). Every stage does an `artifact` upsert, so every
  stage's run-twice idempotency test runs against a real Postgres (testcontainers), per the
  Postgres-not-SQLite rationale in [Database](#database). Separately, render's other concern — the
  pyrender/OSMesa offscreen render — can't run in the host test environment, so its test mocks the GL
  call (`_render_views`) and asserts only the pure view-set + camera-pose math.

## Database

Requirements (resolves the DB TODO):

- **Purpose** — the metadata DB is the source of truth for pipeline state and labels; object
  storage holds the heavy binary artifacts (raw meshes, rendered images/point clouds). The DB
  stores paths/keys into object storage, never the blobs themselves.
- **Core entities:**
  - `model` — store id, source URL, license, download status, content hash, raw object key.
  - `artifact` — per-stage outputs (converted / normalized / rendered), object keys, stage status.
  - `label` — model id, class, source (`weak` | `manual`), confidence, annotator, timestamp.
    Keep weak and manual labels as distinct rows so weak-vs-corrected analysis is possible
    (see [ml.md](../ml/ml.md#evaluation)).
  - `training_run` — id, config snapshot, data version, metrics, artifact keys, status
    (feeds the [dashboard](../web/web.md#training-dashboard)).
  - `user` — id, email, role (`user` | `admin`) — see [web.md](../web/web.md#auth--roles).
- **Access patterns:** filter models by label/status for the browse view (paginated); look up a
  single model + its artifacts + labels for the detail view; aggregate label counts per class
  for metadata exploration and dashboards.
- **Requirements:** relational (foreign keys between model/artifact/label), transactional
  status updates, cheap-to-run (fits the budget), managed if the chosen cloud offers a
  low-cost tier.
- **Decision → PostgreSQL everywhere.** Cloud: managed **Cloud SQL for PostgreSQL**, smallest
  shared-core tier — not AlloyDB (built for far heavier OLTP/analytical load, much costlier) and not
  self-managed Postgres on a VM (needless ops). Local (milestone 2): **Postgres in Docker Compose,
  not SQLite**, so dev and prod share one SQL dialect and concurrency model.
- **Why the same engine in dev — idempotency & scale:**
  - **Idempotency (NFR-2).** The check-output-then-act invariant is implemented as atomic upserts
    keyed on (model id, stage) / content hash — Postgres `INSERT … ON CONFLICT DO NOTHING/UPDATE`
    makes "skip if already processed" race-safe under concurrent redelivery. SQLite's upsert and
    locking semantics differ, so an idempotency test green on SQLite could still race on Cloud SQL;
    running Postgres locally means the run-handler-twice test actually exercises production semantics.
  - **Scale (NFR-3).** Preprocessing is embarrassingly parallel — many workers write status updates
    concurrently. Postgres uses row-level locking + MVCC, so writers to different rows don't block
    each other. SQLite serializes every write behind a single database-level lock (one writer at a
    time), which both hides concurrency bugs in dev and would bottleneck at real worker fan-out.
    Postgres-everywhere keeps dev concurrency faithful to the cloud fan-out we scale to.
- **Implementation.** SQLAlchemy 2.0 ORM in `server/app/models.py` on a shared `Base`
  (`server/app/db.py`), with the connection string and other settings read from `IMAGEGENIE_`-prefixed
  env vars (`server/app/config.py`). Milestone 2 defined `model` (the download stage); milestone 4
  adds `artifact` — one row per (model, stage) output with a unique `(model_uid, stage)` constraint
  backing the idempotent upsert. `label`/`training_run`/`user` land with their stages.

## API Layer

Resolves the earlier "do we use a REST API at all?" question: **yes — exactly one, and only for the
frontend.**

- **A single REST API (FastAPI)** acts as the backend-for-frontend. It serves labels/metadata/results
  to the [browse and detail views](../web/web.md#labeling-ui) and the
  [training dashboard](../web/web.md#training-dashboard), handles [login/auth and roles](../web/web.md#auth--roles),
  accepts admin label corrections (FR-4 / FR-8), and receives admin
  [data uploads](../web/web.md#data-upload) (FR-9), enqueuing them into the pipeline. Pydantic models
  define its request/response schemas.
- **Workers do not use REST to talk to each other.** Inter-stage handoff is Pub/Sub messages, never
  worker-to-worker HTTP. The download stage (Cloud Run *job*) has no HTTP surface at all.
- **Caveat — the push endpoints are not a public API.** Preprocessing Cloud Run *services* do expose
  an HTTP endpoint, because Pub/Sub **push** delivers each message as an HTTP POST — but that is an
  internal push-delivery webhook (a Pub/Sub receiver), not a REST API designed or versioned for
  clients. Keep it off the public API surface: authenticate it as a Pub/Sub push endpoint and don't
  document it as client-facing.

### Endpoints and access control

Implemented in `server/app/api.py`. **Every endpoint except `/healthz` and `/auth/login` requires a
session**; label writes additionally require the `admin` role (FR-8, NFR-7).

| Endpoint | Access | Notes |
|----------|--------|-------|
| `GET /healthz` | public | liveness probe |
| `POST /auth/login` | public | sets the cookie pair; 401 bad credentials, 403 unverified |
| `POST /auth/signup` | public | invite-gated; 400 short password, 403 no invite, 409 email taken |
| `POST /auth/verify-email` | public | consumes a one-time token; 400 invalid/expired |
| `POST /auth/verify-email/resend` | public | re-issues a link; **always 204** |
| `GET /auth/me` | logged in | the caller's email + role |
| `POST /auth/logout` | public | revokes the session server-side; 204 |
| `POST /auth/invites` | **admin** | mints an email-bound invite; idempotent per email |
| `GET /models` | logged in | paginated; filter by `class_name` / `source` |
| `GET /models/{uid}` | logged in | resolves the model's *current* label |
| `PUT /models/{uid}/label` | **admin** | records a manual label, attributed to the calling admin |

- **Sessions, not JWTs.** Login mints an opaque random token stored in the `session` table and
  returned as an **httpOnly** cookie (`imagegenie_session`, 14-day TTL) that page JS can't read.
  Server-side state is the point: logout revokes immediately, which a stateless JWT can't do.
- **Roles are checked at the route.** `current_user` resolves the cookie (401 if absent/expired/
  unknown); `require_admin` layers the role check on top (403). Read routes that don't need the
  caller's identity declare the dependency rather than taking an unused parameter.
- **Corrections are attributed.** `PUT /label` writes `label.annotator = <calling admin's email>`,
  so weak-vs-corrected analysis can tell who changed what.
- Still to come in this area: dead-letter endpoints and admin [data upload](../web/web.md#data-upload)
  (FR-9).

### Serving artifacts

The labeling UI needs the pipeline's output — without it the viewer shows a placeholder and the grid
shows emoji, so nobody can actually label. Key layout lives in one place, `server/app/artifact_keys.py`,
imported by both the workers that write and the API that reads: a key format duplicated between
writer and reader drifts silently, as a missing image rather than an error.

- **`GET /models/{uid}/artifacts`** → `{views: [...], mesh}` for the detail view. It checks each blob
  exists, so a model part-way through the pipeline yields fewer views (or none) and the UI shows a
  placeholder rather than broken images.
- **`ModelSummaryOut.thumbnail`** carries the first view for the grid, and is deliberately emitted
  **without** an existence check — a 24-card page would otherwise cost 24 round-trips to object
  storage just to draw thumbnails. Signing is local and free; the client treats a 404 as "no
  preview". This is the one place we knowingly return a URL that may not resolve.
- **Signed URLs where possible.** The browser reads GCS directly rather than proxying 12 images per
  card through the API, which would make it the bottleneck and pay egress twice. TTL is 15 minutes —
  a signed URL is readable by whoever holds it, so it should outlive a page render and not much else.
- **`GET /artifacts/{key}` streams as a fallback** for backends that can't sign (local dev). Login
  required, since this is the dataset (NFR-7). Note the asymmetry: signed URLs are readable without a
  session until they expire. That is the trade for not proxying, and the reason the TTL is short.
- ⚠️ **Signing goes through IAM on Cloud Run.** The metadata server's credentials have no private
  key, so `GcsStorage.signed_url` signs via the IAM `signBlob` API — it passes the runtime SA's email
  and a live access token to `generate_signed_url` (without those it would try to sign locally and
  always raise). Two things the deploy must provide: the runtime SA holding
  `iam.serviceAccountTokenCreator` **on itself**, and `IMAGEGENIE_SIGNER_SA_EMAIL` set to that SA
  (else it falls back to whatever email the credentials report). If signing fails anyway, the code
  logs a warning and streams the blob through the API instead — the page still works, slower, and the
  log line is the signal the binding is missing.

### Dead letters

Jobs that fail a stage are recorded in a `dead_letter` row **by the worker at nack time**
(`server/app/web.py` → `app/dead_letters.py`), and the admin view reads that table.

The alternative — having the API pull from the `*-jobs-dlq-sub` subscriptions — was rejected:

- **The error text only exists at nack time.** A Pub/Sub dead-letter message carries the original
  payload and a delivery count, never *why* the subscriber gave up. Reading the DLQ can tell you a
  job failed but not what went wrong, which is the one thing an operator needs.
- **Pulling to render a page consumes what it displays.** Messages would have to be nacked back, and
  listing would be slow and non-deterministic.
- **Records outlive Pub/Sub retention** (7 days by default), so a failure from an old ingestion run
  is still visible.

Details:

- **Unique on `(model_uid, stage)`,** upserted. At-least-once delivery means the same job fails
  repeatedly; the admin wants current state, not one row per attempt. Re-failing also clears
  `replayed_at`, so a replayed job that fails again is outstanding once more.
- **No FK to `model`** — a download can fail before the model row exists, and losing the record
  would be worse than the missing referential integrity.
- **Errors are truncated** to 2000 chars; a mesh library traceback can run to kilobytes.
- **Retry republishes to the stage topic** and keeps the row, marked `replayed_at`, so an admin can
  see they already tried. Safe to press freely: every stage is idempotent (NFR-2), so replaying a
  job that already succeeded is a no-op.
- **Admin-only** — operational detail, and retry re-enqueues real pipeline work.
- **Recording never breaks the nack.** Its own failure is caught and logged; a DB hiccup must not
  turn a retryable job into a lost one.
- `app/replay_dlq.py` remains the bulk tool for draining a whole Pub/Sub DLQ back to its topic; this
  table backs the per-item view. The two are complementary — the CLI is what recovered 13,189
  messages after the milestone-4 run.

### Migrations

**Alembic owns the schema** (`server/alembic/`). It replaces the skeleton's `create_all` bootstrap,
which could add a *missing table* but never alter an existing one — so a new column simply never
appeared, and the failure surfaced later as a query error rather than at startup.

```
make migrate                        # alembic upgrade head
make migration MSG="add x to y"     # autogenerate from model changes
make migration-status               # current revision + head
```

- **The URL is never in `alembic.ini`.** `env.py` reads `IMAGEGENIE_DATABASE_URL` (Secret Manager in
  cloud) — the connection string carries the DB password. It only falls back to app config when the
  caller hasn't supplied a URL, so tooling and tests can point at another database without silently
  migrating the default one.
- **`create_all` is off by default** (`IMAGEGENIE_AUTO_CREATE_SCHEMA`). The two must not both create
  tables: if `create_all` wins, the migration that would have created that table fails as "already
  exists" and the version table then disagrees with reality. The local Docker stack sets it true so
  `make compose-up` needs no separate step; deployed environments leave it off.
- **Deploys run `alembic upgrade head` as a step**, not from every instance at startup — with the
  workers at maxScale 10–25, per-instance migration would mean a dozen racing upgrades.
- **A drift test guards the pair** (`tests/test_migrations.py`): it builds a database from migrations
  alone and asserts it matches `Base.metadata`. Nothing else would catch a model change shipping
  without a migration, because the rest of the suite builds its schema with `create_all`.
- **Enum types need explicit drops in `downgrade`.** Autogenerate omits them, and Postgres ENUMs
  outlive the tables that use them, so a downgrade→upgrade cycle fails with "type already exists".
  The initial revision drops all five by hand; **any future revision adding an enum must do the
  same.**

> **Adopting this on the existing Cloud SQL database.** The instance has been written to by the
> workers since milestone 3, and its schema was materialized by `create_all`, which records nothing
> about which revision the schema corresponds to. To Alembic a database with no `alembic_version`
> table is indistinguishable from an empty one, so `upgrade head` starts at revision zero and runs
> the initial migration — which aborts partway through on `CREATE TABLE model` ("already exists"),
> after having created the earlier tables and still without a version row. Skipping the migration
> isn't an option either, because the database genuinely lacks most of what the API needs.
>
> **Verified state (2026-07-22, via `cloud-sql-proxy` + `psql`):**
>
> | | |
> |---|---|
> | Tables | `model`, `artifact` only — no `alembic_version` |
> | Missing | `app_user`, `invite`, `session`, `email_verification`, `label`, `dead_letter`; `model.title`, `model.tags` |
> | Enums | `downloadstatus`, `artifactstage`, `artifactstatus` already exist |
> | Data | 12,009 models (all `downloaded`); 35,469 artifacts — 11,858 converted / 11,826 normalized / 11,785 rendered |
>
> Two ways forward. **Both are viable; the blob store makes the destructive one cheap.**
>
> - **Adopt in place** — hand-apply the delta (six tables, two columns), then `alembic stamp head` to
>   write the version row. Stamping must come *second*: stamp first and the missing tables never get
>   created. Keeps `content_hash` and every row as-is. The cost is a hand-written delta that has to
>   match head exactly, or later `--autogenerate` diffs come out wrong.
> - **Drop and rebuild** — `DROP SCHEMA public CASCADE; CREATE SCHEMA public;` then
>   `alembic upgrade head`. This also clears the three enum types, which would otherwise collide.
>   Yields a schema that matches head exactly, with no hand-written SQL to get wrong. **The row loss
>   is recoverable without re-ingesting**, because every object key embeds its uid
>   (`raw/{uid}.glb`, `processed/converted/{uid}.ply`, `processed/normalized/{uid}.ply`,
>   `processed/renders/{uid}/`) — `make reconcile-storage` rebuilds `model` and `artifact` from a
>   bucket listing alone (see [Rebuilding the tables from storage](#rebuilding-the-tables-from-storage)),
>   and the metadata and weak-label backfills restore the rest.
>
> **Do not drop anything before confirming the blobs are present** — the GCS listing is what makes
> the rows recoverable, and it is the only copy. Verified the same day: 12,029 raw / 11,858 converted
> / 11,826 normalized / 141,420 render PNGs. Converted and normalized match the artifact counts
> exactly, and 141,420 ÷ 12 views = 11,785, exactly the rendered-artifact count — so every row in the
> table above has its blobs behind it. (Raw carries 20 objects with no row, most likely the pilot
> models from before the table was populated; harmless either way.)

### Data upload

`POST /models/upload` (admin-only, FR-9) takes a mesh as `multipart/form-data` and enqueues it.
**The upload stands in for the download stage** — the file *is* the raw mesh — so it lands at
`raw/<uid>.<ext>` and goes straight to the convert topic. After that it is an ordinary model: the
remaining stages, the labeling UI, and the reconciler can't distinguish it from an ingested one.

**Transport: a plain POST through the API**, not a signed resumable upload URL straight to GCS. The
signed-URL route would lift the size ceiling and spare the API the bandwidth, but it depends on the
same signing path that is [currently broken on Cloud Run](#deploying-the-api-to-cloud-run), and it
needs a confirm-then-enqueue round trip to avoid orphaned blobs. A direct POST needs no new IAM and
is bounded anyway by Cloud Run's own 32 MiB request-body ceiling, which `upload_max_bytes` mirrors so
the caller gets our error rather than the platform's.

**Everything unusable is rejected at the door**, because the alternative is a dead-letter minutes
later with nothing tying it back to the admin who uploaded it:

| Condition | Status |
|---|---|
| Extension not in `RAW_SUFFIX_TO_FILE_TYPE` (notably `.fbx`) | `415` |
| Body over `upload_max_bytes` | `413` |
| Empty file | `400` |
| Bytes that trimesh can't load, or a mesh with no faces | `422` |

The mesh parse is the one deliberate cost: it runs `load_mesh` in the request path, with trimesh
imported inside the handler so the API doesn't pull the mesh stack in at startup. Justified because
the route is admin-only and rate-limited, and a corrupt file accepted with `201` would otherwise fail
invisibly three stages downstream. Its error is sanitized before it reaches the client: `load_mesh`'s
own "no geometry" / "no faces" messages describe the file and are passed through, but a raw parser
internal (`"buffer size must be a multiple of element size"`) is logged and replaced with a generic
"could not read this file as GLB" — an admin can act on the latter, not the former.

**Uids are generated (`uuid4().hex`), not derived from content.** Re-uploading the same mesh
therefore creates a second model. Content-addressing would deduplicate, but it would also make two
admins uploading the same file collide on a single row and silently share its labels. The
`title` is taken from the uploaded filename — the uid is random hex, so it is the only
human-readable handle the labeling UI would otherwise have.

Uploaded models carry **no weak label**, since weak labels come from store metadata that an upload
has none of. They appear in the browse grid as unlabeled and are labeled by hand.

### Soft delete

Admins can delete a model (FR-9), but a delete is **soft**: it sets `model.deleted_at` and drops
neither the rows nor the blobs. A deleted model vanishes from every route a labeler uses —
`GET /models`, `GET /models/{uid}`, its artifacts, and label writes all treat it as a 404 — while a
separate Deleted view opts *into* deleted rows so an admin can restore one.

- `DELETE /models/{uid}` — soft-delete (admin, 204). Idempotent: deleting an already-deleted model
  is a no-op, so a double click or retried request doesn't error.
- `POST /models/{uid}/restore` — clear `deleted_at` (admin), returning the now-visible model.
- `GET /models/deleted` — the paginated restore queue, most-recently-deleted first (admin only:
  which models were removed is operational detail). **Registered before `GET /models/{uid}`** so the
  literal path wins — otherwise `deleted` binds as a uid and the view is unreachable.

Soft, not hard, for two reasons. A mistaken delete should be recoverable rather than vaporize
ingestion spend; and it composes with the reconciler. **A hard delete that dropped only the DB rows
would be undone by the next `reconcile-storage`**, which rebuilds any model whose blobs exist — so a
true purge would have to delete the blobs too, the irreversible operation. `deleted_at` lives only in
the DB and the reconciler's upsert writes only storage-authoritative columns, so a soft delete
correctly survives a rebuild (tested in `test_reconcile_from_storage.py`). Reclaiming storage — the
cost guardrail's "delete raw files for excluded models" — is a separate, deliberate step, not wired
to this button.

`_require_live_model` is the single existence check the write and artifact routes share, so a route
can't act on a deleted model by checking only for existence.

### Rebuilding the tables from storage

Object storage is the durable record; `model` and `artifact` are an index over it. Every key carries
its uid (`app/artifact_keys.py`, `uid_from_key`), so `server/app/reconcile_from_storage.py`
(`make reconcile-storage`, `DRYRUN=1`) reconstructs both tables from a bucket listing — no
re-downloading, no re-rendering. That is what makes dropping the schema a cheap way to adopt
migrations, and it doubles as disaster recovery for the database generally.

```
make reconcile-storage DRYRUN=1     # scan and report, write nothing
make reconcile-storage
```

- **Listing only — no blob bodies are read**, so a full pass over ~165k objects costs no egress.
- **Idempotent (NFR-2):** rows are upserted, so a rerun is a no-op and an interrupted run is simply
  restarted. Verified against real Postgres in `tests/test_reconcile_from_storage.py`, because the
  guarantee is entirely `ON CONFLICT` semantics.
- **An incomplete render set is not recorded.** A model with fewer than `NUM_VIEWS` PNGs is counted
  and reported, not written as `done` — the stage-skip gate trusts the row, so recording a partial
  set would permanently hide that model from a re-render and quietly train on missing views.
- **A model whose raw mesh is gone still gets a row**, marked `pending` rather than `downloaded`.
  Raw files are deleted for models excluded from the dataset (the cost guardrail in CLAUDE.md), and
  the labeling UI reads the processed artifacts, not the source mesh.
- **Unrecognised keys are counted, never imported** — a bucket may hold stray objects, and the run
  reports them rather than inventing models from them.

Two columns it cannot restore, both by design:

- **`content_hash`** — the workers store a sha256 of the bytes; the object store keeps its own
  md5/crc32c, a different digest. Recovering it would mean downloading every blob, which is real
  egress against NFR-1. Nothing reads the hash for correctness (the stage-skip gate checks the row
  status and the blob's presence), so a rebuilt table is functionally equivalent but not
  byte-identical here. Existing hashes are coalesced, never overwritten with null.
- **`title` / `tags`** — these come from the store's annotations, not the blobs. Run the metadata
  backfill below afterwards; the reconciler leaves any already present untouched, so the two tools
  compose in either order.

### Metadata backfill

`model.title` / `model.tags` hold the store metadata the labeling UI shows. The download worker
stores the mesh but not the annotations, so `server/app/backfill_metadata.py`
(`make backfill-metadata`, `LIMIT=N` / `DRYRUN=1`) fills them in afterwards.

- **Why it matters:** without it a card's only caption is its uid. Titles are frequently what settles
  the ambiguous classes — `figure` vs `animal` is the weakest at 0.62 precision
  ([ml.md](../ml/ml.md#weak-label-policy)), and the title usually decides it.
- **Categories are folded in with tags.** Both are free-text descriptors that help a labeler; the
  distinction only matters to the weak-labeling rules, which read the annotations directly.
- **Idempotent (NFR-2)** — only rows with no title are fetched, so a rerun after a partial pass
  resumes. Models with no usable annotation are counted, not failed: one bad row must not abort a
  30k-row run.
- **Batch tool, never on-demand.** `objaverse.load_annotations` downloads whichever of the ~160
  shard files contain the requested uids, and our uids are spread across all of them — the first run
  pulls most of the shard set (hundreds of MB, cached afterwards). The API must never do this.
- The API falls back to `model <uid[:8]>` until the backfill runs, so a dull caption rather than none.

> ⚠️ **This adds columns to an existing table, and there is no migration tool.** `init_db` uses
> `create_all`, which creates *missing tables* but never alters existing ones — so new tables (like
> `invite`) appear automatically while new **columns** do not. A database created before this change
> needs the columns added by hand:
>
> ```sql
> ALTER TABLE model ADD COLUMN title TEXT, ADD COLUMN tags TEXT[];
> ```
>
> (Adding nullable columns is instant in Postgres — no table rewrite.) This is the project's second
> schema change against a live DB; **adopting Alembic is overdue** before a third lands.

### Weak-label backfill

Weak labeling (FR-3) writes `data/exploration/weak_labels.csv`, but the labeling UI reads the DB —
so without a load step every model shows as "unlabeled" and there is nothing to confirm or correct.
`server/app/backfill_labels.py` (`make backfill-labels`, `DRYRUN=1` to preview) is that bridge. Run
it after ingestion, and again whenever the weak-labeling rules are re-run.

- **Confidence is the measured per-class precision** from `weak_label_eval.json`
  ([ml.md](../ml/ml.md#evaluation)) — not an invented number. It is literally "how often is a weak
  label of this class correct", graded against the LVIS gold set, which makes
  lowest-confidence-first a meaningful review order: `figure` (0.62, the known figure/animal
  boundary) surfaces ahead of `lamp` (1.00). That is the ordering the active-learning loop
  (milestone 8) wants, so it is worth getting right at load time.
- **Idempotent (NFR-2).** A model that already has a weak label is skipped, so reruns insert
  nothing. Duplicate rows within the CSV also collapse to one.
- **Manual corrections survive a rerun.** They are separate rows and the API resolves the *most
  recent* label as current, so re-importing can't clobber human work — covered by a test, since
  that's the failure that would be worst and quietest.
- **Rows whose model isn't in the DB are skipped, not an error** — the CSV covers the whole labeled
  set while the DB holds only what finished downloading. The run reports the count so the gap is
  visible rather than assumed.
- Loading a 32,777-row CSV against 8,000 downloaded models takes under a second.

### Signup, verification, and invites

Account creation is **invite-only** — there is no open registration, which is what keeps FR-8's
"login required" meaningful on a public URL. An admin mints an email-bound `invite` row; signup
consumes it and creates an *unverified* account; a one-time emailed token flips `verified`; only then
can the account log in.

- **Error ordering on signup is a privacy decision.** The invite is checked *first*, so a caller with
  no invite for an address learns only `invite_required` and can't probe which addresses have
  accounts. `email_taken` is reachable only once an invite exists for that address — i.e. by someone
  who already knows an admin invited it.
- **Resend always returns 204**, whatever the address. A status that varied with account existence
  would be an enumeration oracle on an endpoint reachable without logging in.
- **Verification tokens are stored as SHA-256 hashes**, never in the clear, so a leaked DB snapshot
  doesn't hand out the right to verify accounts. Plain SHA-256 rather than bcrypt is correct here:
  these are 256-bit random values, so there is nothing for a slow hash to defend against.
- **One live token per account.** Issuing deletes any outstanding token, so a resend invalidates the
  previous link and the table can't be grown by repeatedly asking for one.
- **Tokens are consumed even when expired** — a one-time token must not survive its own use. Note the
  ordering this forces: the failure is raised *after* the transaction commits, because raising inside
  `session_scope` rolls the block back and would leave a spent token replayable.
- **Invites never grant admin.** Signup always creates a `user`; promotion is a deliberate manual
  step.
### Email

**Provider → Resend** (`server/app/mail.py`), reached over plain HTTP rather than its SDK — the API
is a single POST, and a dependency wrapping one request isn't worth carrying. Configured by
`IMAGEGENIE_RESEND_API_KEY`, `IMAGEGENIE_MAIL_FROM`, and `IMAGEGENIE_APP_BASE_URL` (the *frontend*
origin, since the links point at the SPA).

- **Sending never breaks a flow.** By the time we send, the account already exists — failing the
  request would strand a created account behind an error. Delivery failures are logged and swallowed;
  the user can request a resend.
- **Queued as a FastAPI background task**, so a slow provider doesn't hold the response open. Tasks
  run only after a successful response, so a rolled-back signup never emails a link for an account
  that doesn't exist.
- **A 10s timeout on the send**, per [Request Resilience](#request-resilience) — a hung provider must
  not stall a worker. There is no retry: a failed send is dropped and recovered by the user hitting
  resend.
- **No API key → log the link instead of sending**, so local dev needs no credentials. This writes a
  token-bearing link into the logs and is therefore strictly a development affordance: **every
  deployed environment must set `IMAGEGENIE_RESEND_API_KEY`.** The Cloud Run deploy enforces this —
  `mail_from` and `resend_api_key` are required tfvars ([api.tf](../infra/api.tf)), so the service
  can't come up without them.
- **The transport is swappable** (`set_mail_sender`) and tests use that seam, so subjects, bodies, and
  generated links are actually asserted. Testing only the no-key path would leave the builder — the
  part that can silently generate a broken link — uncovered.
- **Interpolated values are HTML-escaped** at the boundary, even though addresses are validated
  upstream.
- ⚠️ **Deliverability is unconfigured.** The default `onboarding@resend.dev` is Resend's sandbox
  sender, which delivers **only to the Resend account owner's own address**. Real delivery to invited
  labelers needs a domain we own with SPF/DKIM/DMARC set up and verified in Resend. Until that's
  done, invites only work for the project owner.

### CSRF

Cookie authentication means the browser attaches credentials to *any* request it makes to the API,
including one triggered by another site. Two layers stop that:

1. **`SameSite=Lax`** on the session cookie — blocks the classic cross-site form POST outright.
2. **A double-submit token** — login mints a random token (`secrets.token_urlsafe(32)`) and sets it
   as a *second*, deliberately **non-httpOnly** cookie (`imagegenie_csrf`, same attributes and TTL as
   the session). The client reads it and echoes it in an `X-CSRF-Token` header on every unsafe
   request; the server compares the two with `hmac.compare_digest`. Security rests on the same-origin
   policy: a cross-site page can make the browser *send* the cookie but can neither read its value
   nor set the header.

The token holds **no server-side state** — nothing to store, expire, or replicate — so `SameSite`
remains the primary control and the token is the belt-and-braces layer for fetch-issued requests.

- **Enforced as middleware, not a per-route dependency**, so it **fails closed**: `GET`/`HEAD`/
  `OPTIONS` are exempt as safe methods, and the only exempt path is `POST /auth/login` (it runs
  before a session exists and is what mints the token). A new state-changing endpoint — upload, DLQ
  replay — is protected the day it is added; skipping the check requires deliberately editing
  `CSRF_EXEMPT_PATHS`.
- **Logout is not exempt.** It is a state change, and a cross-site forced logout is exactly the
  nuisance this protects against.
- **The CSRF layer answers before auth**, so an anonymous write gets `403 csrf_failure` rather than
  `401`. Not a UX regression for an expired session: the two cookies share a TTL, and a server-side
  revocation leaves the CSRF cookie in place, so that path still matches and falls through to a 401.
- **`Secure` is config-driven** (`IMAGEGENIE_COOKIE_SECURE`, default off) so local dev works over
  plain http. **Every deployed environment must set it true.**
- **No CORS middleware, deliberately.** The API and the frontend are same-origin; adding permissive
  CORS would undercut both layers above.
- **Not yet done — CSP.** Because this scheme rests on the same-origin policy, an XSS defeats it. A
  strict Content-Security-Policy is the complementary control and is not yet configured.
- **Frontend still to wire.** `web/src/api/` is currently a mock; the real client needs one fetch
  wrapper that attaches the header for any method outside `GET`/`HEAD`/`OPTIONS`
  (see [web.md](../web/web.md#auth--roles)).

### Rate limiting

Implemented in `server/app/ratelimit.py`. Two primitives, because they answer different threats:

| Surface | Key | Limit |
|---------|-----|-------|
| `POST /auth/login` | IP | 20 / 10 min (volumetric) |
| `POST /auth/login` | account | exponential backoff — 3 free, then 1s→15 min doubling |
| `POST /auth/signup` | IP | 10 / 10 min |
| `POST /auth/verify-email` | IP | 20 / 10 min |
| `POST /auth/verify-email/resend` | IP **and** email | 5 / 10 min each |
| `POST /auth/invites` | admin id | 50 / 10 min |
| `PUT /models/{uid}/label` | user id | 600 / 10 min |

- **Login is the endpoint that matters**, and not only for guessing: bcrypt is expensive *by design*,
  so an unthrottled login is a CPU-exhaustion lever as much as a credential-grinding one. Both checks
  therefore run **before** the DB read and before hashing — while locked out the server does no work.
- **Backoff, not a volumetric cap, for the per-account limit.** A volumetric cap counts successes too
  and never escalates. Backoff keys on *failures*: an honest user who mistypes clears the streak by
  logging in, while an attacker grinding one account waits geometrically longer. The lockout arms on
  the failure that crosses the grace window, so it takes effect on the *next* attempt.
- **The per-IP cap covers what backoff can't.** Backoff is per-account, so an attacker sweeping many
  usernames never trips it; the volumetric per-IP cap bounds that sweep.
- **A correct password on an unverified account is not a failure** and must not feed the ladder — it
  isn't a guess.
- **Label writes are a runaway guard, not an abuse control.** Admins are trusted; the cap exists
  because every `PUT` inserts a `label` row, so a looping frontend would grow the table without
  bound. Set far above human labeling speed so it cannot interrupt a real session.
- **Resend is capped on both dimensions** because it triggers an outbound email: per IP (one host
  spraying) and per address (mailbox-bombing a single victim).
- **No token endpoint is left uncapped.** Verification tokens are 256-bit random so guessing is
  hopeless, but an unthrottled token endpoint is still a free oracle.
- **429 always carries `Retry-After`** so clients wait rather than hammer.
- **Fixed window, deliberately.** Its known weakness — a burst straddling a boundary can briefly
  reach 2x the cap — is irrelevant for volumetric caps set this generously; the case where precision
  would matter is login, which uses backoff instead.
- **Per-IP keying trusts `X-Forwarded-For` only when told to** (`IMAGEGENIE_TRUST_PROXY_HEADERS`,
  default off). Believing the header when the app is *not* behind a proxy lets a caller rotate it per
  request and walk around every per-IP cap. Cloud Run deployment must turn it on.

**Known limit — the counters are per-process and in-memory.** That is correct today because the API
is not yet deployed (only workers are in Terraform), so it runs as a single instance. Deploying it
with `max_instance_count > 1` silently multiplies every cap by the instance count and splits the
backoff state. Before that ships, either pin the API to one instance or move the counters into a
shared store. There is no Redis in this project and adding one costs standing spend against the $100
budget (NFR-1), so pinning is the cheaper answer unless the API needs to scale.

### Deploying the API to Cloud Run

Goes to **Cloud Run in the same project as the workers**, because the API talks to Cloud SQL and GCS
constantly and hosting it elsewhere would mean exposing Cloud SQL publicly and paying cross-cloud
egress on every read (see the Open Decision in [CLAUDE.md](../CLAUDE.md)). Cloud Run gives a free
HTTPS `*.run.app` URL; a custom domain can come later.

**Provisioned by `infra/api.tf`** — the `imagegenie-api` service (its own least-privilege SA), public
(`allUsers` invoker; the app gates itself behind login), running the shared image with the API
entrypoint. It wires the four things that would otherwise break the deploy:

1. **Pinned to one instance** (`max_instance_count = 1`). Rate-limit counters are per-process and
   in-memory (see the note above); more than one instance multiplies every cap and splits login
   backoff. Pinning is cheaper than the alternative — a shared store means Redis, standing spend
   against NFR-1's $100.
2. **Signing wired** — the SA holds `iam.serviceAccountTokenCreator` on itself and
   `IMAGEGENIE_SIGNER_SA_EMAIL` is set to it, so `GcsStorage.signed_url`'s IAM `signBlob` path works
   (the code fix is done; [Serving artifacts](#serving-artifacts)). This is the one thing only real
   GCS on Cloud Run can fully confirm, so **check the logs after the first deploy** for the "could
   not sign a URL … falling back to streaming" warning — its absence is the proof.
3. **The SPA is in the image** (`/srv/web_dist`, `IMAGEGENIE_SPA_DIR` baked), so the service just runs
   `uvicorn app.api:root_app` and serves both the SPA and API on one origin
   ([Serving the SPA](#serving-the-spa)).
4. **The HTTPS/proxy env is set** — `IMAGEGENIE_COOKIE_SECURE=true` (the session cookie must be
   `Secure`) and `IMAGEGENIE_TRUST_PROXY_HEADERS=true` (Cloud Run is a proxy; without it every request
   keys to the same front-end IP for rate limits).

**The one manual gate is the schema.** The existing Cloud SQL database was built by the workers'
`create_all` and knows nothing about migrations — the API cannot serve a single authenticated request
until the auth tables exist. `scripts/adopt_schema.sh` does the drop-and-rebuild end to end: it
verifies the buckets are populated, drops the schema, runs `alembic upgrade head`, rebuilds
`model`/`artifact` from storage ([Rebuilding the tables from storage](#rebuilding-the-tables-from-storage)),
backfills metadata + weak labels, and bootstraps the first admin (`app.create_admin` — signup is
invite-gated and invites need an admin, so a fresh database has no other way in). It is destructive by
design and safe only because the buckets hold the artifacts it rebuilds from — which it checks first.

**Two teardown depths.** The default `adopt_schema.sh` rebuilds the DB *from* the buckets and leaves
them untouched — the buckets are the backup, so this is a non-destructive-to-data reset. For a full
clean slate there is `scripts/wipe_buckets.sh` (the inverse of the reconciler): it deletes the objects
under `raw/` and/or `processed/` — dry-run by default, `--apply` to delete, gated on typing `delete all
objects`. Because those objects are the *only* copy, wiping `raw/` means re-running M4 ingestion to
restore it; `--processed` is cheap (a re-render). After a wipe the populated-bucket gate would (rightly)
refuse the normal path, so pair it with `scripts/adopt_schema.sh --fresh`, which skips that gate, the
reconcile, and both backfills — leaving drop → migrate → admin against empty buckets. The two scripts
stay separate on purpose: two explicit destructive acts with distinct confirmation phrases (`delete all
objects` vs `drop and rebuild`) so neither is reachable by reflex.

**Deploy order:**

1. `make deploy-image` — build + push the image (now including the SPA).
2. `scripts/adopt_schema.sh` — reconcile the schema and seed the admin (once).
3. `terraform -chdir=infra apply` — create the service; prints `api_url`. Also shows a benign in-place
   update on the four worker services (a provider cosmetic, `min_instance_count 0 → null`).
4. `scripts/check_deploy.sh` — health, plus the signing-fallback log scan that confirms gotcha 2.

**Email is required** — `mail_from` and `resend_api_key` are mandatory tfvars, because the deployed
app must be able to send verification and invite mail or nobody but the seeded admin can get an
account (server.md#email). The one exception is `app_base_url` (the link host): the Cloud Run URL
isn't known until the service exists, so leave it empty for the first apply, then set it to the
`api_url` output and re-apply.

### Serving the SPA

The built SPA and the API share one origin (gotcha 4 above). The catch is that the SPA's own
client-side routes share the API's URL namespace — **`/models/{uid}` is both a labeling page and a
JSON endpoint**, and `/dead-letters` likewise. One route can't answer both a browser navigation (wants
the app shell) and a `fetch` (wants JSON), so they can't both live at the root.

**The API therefore mounts under `/api`, and the SPA is served at the root.** `/api` is exactly the
prefix the frontend already sends and the Vite dev server already strips, so nothing in the client
changes between dev and prod.

- The API app (`app`, every route unprefixed) is **untouched**: it still runs at the root under
  `uvicorn app.api:app` for local backend-only dev, and the whole test suite exercises it that way.
- `app.api:root_app` is the **production entrypoint**: it mounts `app` at `/api` and serves
  `IMAGEGENIE_SPA_DIR` at `/` via a catch-all. The catch-all (not a `StaticFiles` mount) returns a
  real file when the path names one and otherwise falls back to `index.html`, because a client-routed
  deep link like `/deleted` is not a file on disk and must reach the browser router. Hashed
  `assets/*` get an immutable cache header; the shell gets `no-cache` so a deploy can't strand
  browsers on a stale `index.html` pointing at asset hashes the new build dropped.
- **The image carries the build.** `server/Dockerfile` is multi-stage — a `node` stage runs
  `npm ci && npm run build`, and the runtime stage copies `web/dist` to `/srv/web_dist` and bakes
  `IMAGEGENIE_SPA_DIR`. Because both `server/` and `web/` must be in the build context, the build
  runs from the **repo root** (`make deploy-image` uses `-f server/Dockerfile .`, and
  `docker-compose.yml` sets `context: ..`). Layer caching skips `npm ci` unless the lockfile changes,
  so iterating on the workers doesn't pay for the SPA build. The same image runs the workers, which
  never read `web_dist`.

Two mount-specific details, both easy to get wrong and both regression-tested:

- **The CSRF middleware matches the mount-relative path.** It runs at the outermost layer, before the
  router strips the mount prefix, so under `/api` the exempt `/auth/login` arrives as
  `/api/auth/login` with `root_path == "/api"`. It strips `root_path` before checking the exempt set,
  or every login would 403.
- **Streaming-fallback artifact URLs carry the mount prefix.** When the backend can't sign a URL
  (local storage, or Cloud Run before the [signing fix](#deploying-the-api-to-cloud-run)), it hands
  the browser a `/artifacts/{key}` path to stream from. Mounted, that has to be `/api/artifacts/...`
  or the browser requests it at the root and gets the SPA shell instead of the blob — so the URL is
  built from the request's `root_path` (empty unmounted, `/api` mounted).

## Request Resilience

Every HTTP request — outbound to the Objaverse API, Pub/Sub push into worker services, and the
frontend's calls to the REST API — follows a common resilience policy so transient failures neither
lose work nor hammer dependencies:

- **Retry with exponential backoff + jitter.** Retry only *transient* failures (connection errors,
  timeouts, HTTP 429 and 5xx) with exponentially increasing, jittered delays. Never retry
  non-retryable 4xx (e.g. 400 / 404) — surface those as errors immediately.
- **Respect `Retry-After` / rate-limit headers.** When the model store returns 429 or a `Retry-After`,
  honour the stated delay rather than guessing — this is the "polite" in FR-1's polite rate limiting.
- **Client-side rate limiting.** Download workers throttle outbound calls to Objaverse (bounded
  concurrency + a request-rate cap) to stay under the store's limits and never burst like a scraper.
- **Timeouts on every request.** Connect + read timeouts so a hung dependency can't stall a worker
  indefinitely; a timeout counts as a retryable failure.
- **Bounded retries → dead-letter.** After a capped number of attempts a job stops retrying and goes
  to the [dead-letter topic](#queue), with the error recorded in the DB — no infinite retry loops.
- **Retries are safe because handlers are idempotent** (NFR-2): a retried download / convert / render
  produces no duplicate work.
- **REST API side.** The FastAPI layer applies [rate limiting](#rate-limiting) and returns proper status codes;
  the frontend retries transient 5xx with the same backoff-and-jitter policy.

## Running the Pipeline Locally (milestones 2 + 4)

The local pipeline is one worker per stage fed by per-stage queues, wired in
`server/docker-compose.yml` (Postgres + Pub/Sub emulator + a `download`, `convert`,
`normalize`, and `render` service, all on the same image) and driven by Makefile
targets:

```
make compose-up               # build + start Postgres, Pub/Sub emulator, all stage workers
make compose-seed COUNT=100   # publish N download jobs (producer)
make compose-down             # stop + remove volumes
make test                     # unit + integration tests (Postgres via testcontainers)
```

Flow: `seed` (producer) publishes `{"uid"}` jobs to `download-jobs` → each stage
consumes its subscription (pull, locally), writes its blob(s) to the `storage`
volume, upserts its DB row (`model` for download, `artifact` for the rest), and
publishes the next stage's job — `download → convert → normalize → render`. Every
stage is idempotent, so re-seeding the same uids re-processes nothing. The image is
the same one that runs on Cloud Run; only the env (managed Pub/Sub, Cloud SQL, GCS)
and delivery mode (push in prod) change.

## Coding Standards (backend)

- **Language:** Python 3.11+. Type hints on all public functions; check with a static type checker.
- **Framework:** FastAPI for the API; Pydantic models for request/response and job payloads.
- **Style:** format + lint with Ruff (or black + ruff). No unformatted code committed.
- **Structure:** each worker type is its own module with a single `process(job)` entrypoint;
  shared storage/DB access behind thin client modules — no direct SDK calls scattered in workers.
- **Config:** all cloud identifiers, bucket names, and credentials via environment variables /
  secrets — never hardcoded.
- **Idempotency first:** every handler is written check-output-then-act. Add a test that runs the
  handler twice and asserts no duplicate work.
- **Logging:** structured logs with model id + stage on every message; errors recorded to the DB,
  not just stdout.
- **Tests:** unit-test conversion/normalize logic on tiny fixture meshes; integration-test the
  queue path on the ~100-model set before scaling.
