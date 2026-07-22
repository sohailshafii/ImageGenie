# Web / Frontend — ImageGenie

The web app: the labeling tool, the training dashboard, auth/roles, and data upload. See
[../CLAUDE.md](../CLAUDE.md) for the project hub and [server.md](../server/server.md) for the API it talks to.

## Scope

- Render 3D models in-browser (three.js).
- Confirm/correct candidate labels with minimal friction (one-keystroke where possible).
- Surface low-confidence examples for hand-labeling (active learning — milestone 8).
- Show training runs and their detail.
- Gate everything behind login; restrict corrections + uploads to admins.

## Labeling UI

Two views, and labels can be corrected in **either** one (resolves the labeling-UI TODO):

### Browse view
- All items visible, **paginated**. Thumbnail grid (rendered multi-view preview per model).
- Inline label confirm/correct without leaving the page — for fast sweeps over many models.
- Filter/sort by class, label source (weak vs. manual), and confidence.

### Detail view
- A single model, full three.js interactive viewer. It loads the pipeline's **normalized PLY**
  (`PLYLoader`) from [the artifacts endpoint](../server/server.md#serving-artifacts) — one download
  per model opened, after which orbiting is entirely client-side. Because the normalize stage already
  centers the mesh and scales its largest extent to 1, the camera framing is fixed and needs no
  per-model fitting. Pipeline PLYs carry no normals, so the viewer computes them; without that the
  mesh renders flat and unreadable.
- The mesh is fetched **separately from the model summary**, so the label panel is usable
  immediately rather than waiting on geometry. A model with no mesh yet shows "No 3D mesh for this
  model yet" — normal for anything the pipeline hasn't normalized, not an error.
- **Dev needs its own proxy entry for `/artifacts`.** Artifact URLs arrive from the server as
  absolute paths and are used raw (in `<img src>` and the loader), so they never pass through the
  client's `/api` prefix. Production serves the SPA and API on one host, where this resolves
  naturally; `vite.config.ts` reproduces that locally.
- Candidate label with confidence; confirm/correct here too.
- Neighboring metadata (store tags/title) shown to aid the labeling decision.

Both views write label corrections through the same API endpoint (see
[server.md](../server/server.md#database) — corrections create/update `label` rows with `source = manual`).

## Training Dashboard

Resolves the dashboard TODO.

- **List view** — all training runs: id, date, config summary, headline metrics, status.
  Sortable/filterable.
- **Detail view (per run)** — full config snapshot, metric curves (loss/accuracy over epochs),
  per-class precision/recall, confusion matrices (see [ml.md](../ml/ml.md#evaluation)), and links to
  artifacts. Backed by the `training_run` entity in [server.md](../server/server.md#database).

## Auth & Roles

Resolves the login TODO.

- **Login required** for all access.
- **Roles:**
  - **Normal user** — read-only: browse models, view labels, view the dashboard.
  - **Admin** — everything a user can do, plus correct annotations and upload data.
- Enforce authorization on the **server** (API layer), not just by hiding UI — the frontend
  role checks are for UX, the backend checks are the security boundary (NFR-7).
- **Account flows (modeled on the ChatApp reference):** signup is **invite-only** — an admin mints an
  email-bound invite, and signup is gated to invited emails; a new account is **unverified** until the
  emailed confirmation link is clicked, with a **resend confirmation** path; login surfaces the
  `unverified` state. Endpoints respond generically (no account enumeration).
- **Implemented (milestone 5), now against the real FastAPI backend:** login, invite-gated signup,
  email verification + resend, and the admin invite UI — see `web/src/api/` (typed client),
  `web/src/auth/` (context + route guards), and `web/src/pages/`. The in-memory mock has been
  removed; swapping it out needed no component changes, which was the point of the single-client rule.
- **How the client talks to the API** (`web/src/api/client.ts`) — one `fetch` wrapper owns the three
  cross-cutting concerns so no caller repeats them:
  - `credentials: 'same-origin'`, so the httpOnly session cookie rides along. Nothing in the app
    reads or stores a token.
  - The **CSRF header** (`X-CSRF-Token`) on any method outside `GET`/`HEAD`/`OPTIONS`, copied from
    the readable `imagegenie_csrf` cookie (see [server.md](../server/server.md#csrf)).
  - Mapping a non-2xx body to a typed `ApiError` code, falling back to the status when the body
    isn't a code it recognizes — so an unexpected response can never surface as a bogus code.
- **Same-origin is a requirement, not a convenience.** The dev server proxies `/api` to the backend
  (`vite.config.ts`) specifically so the browser sees one origin: the cookies are `SameSite=Lax` and
  the CSRF defense rests on the same-origin policy, so a cross-origin setup would need CORS and would
  weaken exactly that. Production must serve the SPA and API behind one host for the same reason.
- **Labels are nullable in the UI.** A model has no label until weak labeling or a human assigns one,
  and the API reports that rather than inventing a class. The grid and detail view render it as
  "unlabeled" with a "— pick a class —" placeholder, and hide Confirm (there is nothing to confirm).
  This is the state *every* model is in until the weak-label backfill runs.
- **One mock remains:** the dead-letter list (`catalog.ts`). The backend has no DLQ endpoints yet —
  `app/replay_dlq.py` is a CLI tool — so `DeadLettersPage` still renders fixed sample rows. The
  function signatures are already the ones the real calls will use.

### Content-Security-Policy (TODO — not yet configured)

The API's [CSRF defense](../server/server.md#csrf) rests on the same-origin policy: an attacker who
can run script on our origin can read the CSRF cookie and forge any request. **A strict CSP is the
complementary control, and it is not in place.**

It belongs here rather than in the API: CSP is enforced on the **HTML document** response, so it is
set wherever the built SPA is served — the API only returns JSON. (A `default-src 'none'` on API
responses is cheap defense-in-depth, but it is not the real control.)

The app is well-positioned for a strict policy — `web/index.html` loads an external module script
with no inline `<script>`, and there is no `eval` or `dangerouslySetInnerHTML` anywhere in `web/src`.
Three things to settle when it lands:

- **Dev and prod need different policies.** Vite's dev server injects an inline HMR script and uses
  `eval`; the production build does neither. The strict policy targets the built output.
- **`style-src` is the friction point** — it governs `style` *attributes* too, so the one remaining
  `style={{…}}` prop breaks under `style-src 'self'`. Rewrite it as a class (cheapest at one
  occurrence) rather than weakening the policy with `'unsafe-inline'`.
- three.js is unaffected either way; shaders are not JavaScript.

Most valuable **before** the SPA is first deployed, not after.

## Data Upload

Resolves the upload TODO.

- Admins can upload additional models into the pipeline.
- Uploaded models enter the same ingestion chain (convert → normalize → render → label) and are
  subject to the same idempotency rules ([server.md](../server/server.md#queue--workers)).
- Validate format (STL/OBJ/GLB/FBX) and size on upload; reject unsupported files with a clear error.

## Coding Standards (frontend)

- **Stack (chosen):** **React + TypeScript + Vite**, three.js for 3D rendering. TypeScript for typed
  model/label data and three.js APIs; Vite for fast dev/build. Lives in `web/`.
- **Auth is a UX layer, not the boundary.** The frontend gates views behind login and hides
  admin-only actions by role, but this is for UX only — the server API is the security boundary
  (NFR-7). Until the FastAPI backend exists, the frontend runs against a typed **mock API** (a single
  swappable client module), so its login/roles are simulated; real enforcement lands with the backend.
- **Rendering:** all 3D viewing through a single reusable viewer component wrapping three.js —
  browse thumbnails and the detail viewer share it. Dispose of GPU resources on unmount.
- **API access:** one typed client module for the FastAPI backend; no fetch calls scattered
  through components.
- **Auth:** never trust the client for authorization; treat role state as a UX hint only.
- **Accessibility & speed:** the browse view must stay responsive with thousands of paginated
  items — virtualize/lazy-load thumbnails; never load the full dataset at once.
- **Formatting/lint:** Prettier + ESLint; no unformatted code committed.
