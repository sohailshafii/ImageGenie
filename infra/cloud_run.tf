# The download worker on Cloud Run (server.md#compute). Runs the same image as the
# local skeleton, but with the command overridden to the uvicorn push receiver
# (server/app/web.py) and storage pointed at GCS. Scales to zero — free at idle.

# Least-privilege runtime identity for the worker.
resource "google_service_account" "worker" {
  account_id   = "imagegenie-worker"
  display_name = "ImageGenie download worker"
}

resource "google_project_iam_member" "worker_cloudsql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_storage_bucket_iam_member" "worker_raw" {
  bucket = google_storage_bucket.raw.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_secret_manager_secret_iam_member" "worker_db_secret" {
  secret_id = google_secret_manager_secret.database_url.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_cloud_run_v2_service" "download" {
  name                = "download-worker"
  location            = var.region
  deletion_protection = false

  template {
    service_account = google_service_account.worker.email

    scaling {
      min_instance_count = 0 # scale to zero
      max_instance_count = 3
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/imagegenie/worker:latest"
      # Override the image's default (local pull worker) with the push receiver.
      # $$ escapes so the shell (not Terraform) expands Cloud Run's $PORT.
      command = ["sh", "-c", "uvicorn app.web:app --host 0.0.0.0 --port $${PORT:-8080}"]

      env {
        name  = "IMAGEGENIE_STAGE"
        value = "download"
      }
      # Publish downstream jobs to the real project's topics (config defaults to the
      # local emulator project); download hands off to the convert stage.
      env {
        name  = "IMAGEGENIE_PUBSUB_PROJECT"
        value = var.project_id
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
        name = "IMAGEGENIE_DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.database_url.secret_id
            version = "latest"
          }
        }
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
    }

    # Mount the Cloud SQL unix socket (connector) into the container.
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
