# Training (M6 chunk G, server.md#training-gpu): the identity that builds the CUDA
# image and the identity the Vertex custom job runs as.
#
# The build needs its own service account because this project has no
# `<number>@cloudbuild.gserviceaccount.com`: Google stopped auto-creating that
# legacy default for newer projects, so a build submitted without
# `--service-account` has no identity to run as and fails with PERMISSION_DENIED
# before it starts. Creating a dedicated one also matches how every other
# identity here is scoped (worker / api / pubsub-push), rather than borrowing the
# broadly-privileged Compute Engine default.

resource "google_service_account" "build" {
  account_id   = "imagegenie-build"
  display_name = "ImageGenie Cloud Build (training image)"
}

# Builds must be able to write their own logs; without this the build fails at
# start with a logging error rather than anything about the Dockerfile.
resource "google_project_iam_member" "build_logs" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.build.email}"
}

# Pull the base image layers and push the finished training image.
resource "google_artifact_registry_repository_iam_member" "build_writer" {
  location   = google_artifact_registry_repository.images.location
  repository = google_artifact_registry_repository.images.repository_id
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.build.email}"
}

# Read the uploaded source tarball.
resource "google_project_iam_member" "build_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.build.email}"
}

# A log bucket we own, rather than the one Cloud Build creates for itself.
#
# A build running as a user-specified service account needs `roles/storage.admin`
# on its log bucket. Granting that on Cloud Build's own auto-created bucket is
# circular — the service only creates it on a first *successful* build, which
# cannot happen while the permission is missing — and granting storage.admin at
# project scope to fix that would hand a build identity full control of the raw
# and processed data buckets. Owning the bucket sidesteps both.
resource "google_storage_bucket" "build_logs" {
  name                        = "${var.project_id}-build-logs"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true # logs are disposable; never block a destroy

  # Build logs are read while debugging a build and never again.
  lifecycle_rule {
    condition { age = 30 }
    action { type = "Delete" }
  }
}

resource "google_storage_bucket_iam_member" "build_logs_admin" {
  bucket = google_storage_bucket.build_logs.name
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.build.email}"
}

# ── The training job's own identity ─────────────────────────────────────────
# Separate from the worker SA: this one never touches Pub/Sub or the raw bucket,
# and the worker never needs Cloud SQL through the connector.

resource "google_service_account" "trainer" {
  account_id   = "imagegenie-trainer"
  display_name = "ImageGenie Vertex AI training job"
}

# Reaches Cloud SQL through the Python connector, which authenticates over IAM
# against the Cloud SQL Admin API — no VPC, and no authorized-network entry for
# an egress IP that Vertex reassigns per job.
resource "google_project_iam_member" "trainer_cloudsql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.trainer.email}"
}

# Reads the rendered views and writes the weights checkpoint back.
resource "google_storage_bucket_iam_member" "trainer_processed" {
  bucket = google_storage_bucket.processed.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.trainer.email}"
}

# The DB URL is fetched from Secret Manager at startup rather than passed as an
# env var, since a Vertex job's environment is visible in its metadata.
resource "google_secret_manager_secret_iam_member" "trainer_database_url" {
  secret_id = google_secret_manager_secret.database_url.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.trainer.email}"
}

# Vertex runs the job *as* this account, which requires the caller submitting the
# job to be able to act as it.
resource "google_service_account_iam_member" "trainer_user" {
  service_account_id = google_service_account.trainer.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.trainer.email}"
}

output "build_service_account" {
  value       = google_service_account.build.email
  description = "Service account `make train-image` submits Cloud Build as."
}

output "trainer_service_account" {
  value       = google_service_account.trainer.email
  description = "Service account the Vertex training job runs as."
}

# ── Launching a run from the dashboard (web.md#starting-a-training-run) ──────
# The API submits the Vertex job itself, which needs two distinct grants: the
# right to create jobs, and the right to *act as* the account the job runs under.
# The second is the one that is easy to miss — without it Vertex refuses with a
# permission error naming the trainer SA, not the API's.

resource "google_project_iam_member" "api_vertex" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_service_account_iam_member" "api_runs_as_trainer" {
  service_account_id = google_service_account.trainer.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.api.email}"
}
