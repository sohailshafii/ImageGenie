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
for per-model preprocessing.

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

- `imagegenie-raw` — downloaded meshes.
- `imagegenie-processed` — converted / normalized / rendered outputs, separated by prefix.

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
directly. The milestone-2 skeleton uses `LocalStorage` over a local directory; a `GcsStorage` with the
same interface swaps in for cloud, so worker code is unchanged between local and GCS.

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
  env vars (`server/app/config.py`). Milestone 2 defines `model` first (the download stage);
  `artifact`/`label`/`training_run`/`user` land with their stages.

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
- **REST API side.** The FastAPI layer applies per-user rate limiting and returns proper status codes;
  the frontend retries transient 5xx with the same backoff-and-jitter policy.

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
