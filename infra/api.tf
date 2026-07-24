# The backend-for-frontend API on Cloud Run (server.md#deploying-the-api-to-cloud-run).
# Runs the shared image with the API entrypoint (uvicorn app.api:root_app), which
# serves the built SPA and the JSON API on one origin. Public — the app enforces its
# own login — and pinned to ONE instance because the rate-limit counters are
# per-process (server.md#rate-limiting). Scale-to-zero, so free at idle.

# --- Least-privilege runtime identity for the API ---
resource "google_service_account" "api" {
  account_id   = "imagegenie-api"
  display_name = "ImageGenie API"
}

resource "google_project_iam_member" "api_cloudsql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.api.email}"
}

# Admin uploads write meshes to the raw bucket (FR-9), so read+write there.
resource "google_storage_bucket_iam_member" "api_raw" {
  bucket = google_storage_bucket.raw.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.api.email}"
}

# The API only reads processed artifacts (renders, meshes) — to serve and to sign
# them — so read-only on the processed bucket.
resource "google_storage_bucket_iam_member" "api_processed" {
  bucket = google_storage_bucket.processed.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"
}

resource "google_secret_manager_secret_iam_member" "api_db_secret" {
  secret_id = google_secret_manager_secret.database_url.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.api.email}"
}

# Admin uploads enqueue a convert job — the upload stands in for the download stage.
resource "google_pubsub_topic_iam_member" "api_convert_publisher" {
  topic  = google_pubsub_topic.stage["convert"].name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${google_service_account.api.email}"
}

# Signing GCS URLs via IAM signBlob: the API SA impersonates ITSELF, which needs
# token-creator on itself (server.md#serving-artifacts). Without this the signer
# fails and the API streams every image instead — slower, not broken.
resource "google_service_account_iam_member" "api_self_sign" {
  service_account_id = google_service_account.api.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:${google_service_account.api.email}"
}

# --- Optional transactional email (Resend) ---
# Left unset, the app logs verification / invite links instead of sending them —
# fine for the pre-seeded admin, but signup for anyone else needs a key
# (server.md#email). The key is a Secret Manager secret so it never sits in the
# service's plain env; the value comes from the (sensitive) tfvar.
resource "google_secret_manager_secret" "resend_api_key" {
  count     = var.resend_api_key != "" ? 1 : 0
  secret_id = "imagegenie-resend-api-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.enabled]
}

resource "google_secret_manager_secret_version" "resend_api_key" {
  count       = var.resend_api_key != "" ? 1 : 0
  secret      = google_secret_manager_secret.resend_api_key[0].id
  secret_data = var.resend_api_key
}

resource "google_secret_manager_secret_iam_member" "api_resend_secret" {
  count     = var.resend_api_key != "" ? 1 : 0
  secret_id = google_secret_manager_secret.resend_api_key[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.api.email}"
}

resource "google_cloud_run_v2_service" "api" {
  name                = "imagegenie-api"
  location            = var.region
  deletion_protection = false

  template {
    service_account = google_service_account.api.email

    scaling {
      min_instance_count = 0 # scale to zero — free at idle
      max_instance_count = 1 # per-process rate limits (server.md#rate-limiting)
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/imagegenie/worker:latest"
      # Override the image's default (download worker) with the API entrypoint,
      # which serves the SPA + JSON API. $$ escapes so the shell expands $PORT.
      command = ["sh", "-c", "uvicorn app.api:root_app --host 0.0.0.0 --port $${PORT:-8080}"]

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi" # trimesh loads on upload; the SPA is static
        }
      }

      env {
        name  = "IMAGEGENIE_STORAGE_BACKEND"
        value = "gcs"
      }
      env {
        name  = "IMAGEGENIE_RAW_BUCKET"
        value = google_storage_bucket.raw.name
      }
      env {
        name  = "IMAGEGENIE_PROCESSED_BUCKET"
        value = google_storage_bucket.processed.name
      }
      # Upload publishes convert jobs to the real project's topics (config defaults
      # to the local emulator project).
      env {
        name  = "IMAGEGENIE_PUBSUB_PROJECT"
        value = var.project_id
      }
      # Behind Cloud Run's HTTPS front end: the session cookie must be Secure, and
      # per-IP rate limits must trust X-Forwarded-For (it IS behind a proxy now).
      env {
        name  = "IMAGEGENIE_COOKIE_SECURE"
        value = "true"
      }
      env {
        name  = "IMAGEGENIE_TRUST_PROXY_HEADERS"
        value = "true"
      }
      # Sign GCS URLs as this SA (server.md#serving-artifacts) rather than relying
      # on whatever email the metadata server reports.
      env {
        name  = "IMAGEGENIE_SIGNER_SA_EMAIL"
        value = google_service_account.api.email
      }
      env {
        name = "IMAGEGENIE_DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.database_url.secret_id
            version = "latest"
          }
        }
      }

      # Optional email config — only wired when the corresponding tfvar is set.
      dynamic "env" {
        for_each = var.mail_from != "" ? [1] : []
        content {
          name  = "IMAGEGENIE_MAIL_FROM"
          value = var.mail_from
        }
      }
      dynamic "env" {
        for_each = var.app_base_url != "" ? [1] : []
        content {
          name  = "IMAGEGENIE_APP_BASE_URL"
          value = var.app_base_url
        }
      }
      dynamic "env" {
        for_each = var.resend_api_key != "" ? [1] : []
        content {
          name = "IMAGEGENIE_RESEND_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.resend_api_key[0].secret_id
              version = "latest"
            }
          }
        }
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
    }

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.main.connection_name]
      }
    }
  }

  depends_on = [
    google_project_service.enabled,
    google_secret_manager_secret_version.database_url,
  ]
}

# Public app: anyone can reach the HTTP endpoint, and the app gates everything
# behind login (only /healthz, the pre-auth flows, and the static SPA are open).
# A public website with application-level auth, not a private service.
resource "google_cloud_run_v2_service_iam_member" "api_public" {
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "api_url" {
  description = "Public URL of the labeling app + API."
  value       = google_cloud_run_v2_service.api.uri
}
