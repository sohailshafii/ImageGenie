# Preprocessing stages (convert → normalize → render) on Cloud Run, mirroring the
# download stage (cloud_run.tf + pubsub.tf) but fanned out with for_each. Each stage
# is a scale-to-zero push service running the shared image with IMAGEGENIE_STAGE set
# (web.py dispatches on it), plus its own Pub/Sub topic + push subscription + DLQ.
# Reuses the worker + push service accounts and the two GCS buckets. Topics/subs and
# idle (scaled-to-zero) services cost nothing — no new always-on spend.

locals {
  preprocessing_stages = toset(["convert", "normalize", "render"])

  # Render rasterizes in software (OSMesa/llvmpipe) and loads whole meshes, so it
  # gets more CPU/RAM than the lighter mesh-IO stages.
  stage_resources = {
    convert   = { cpu = "1", memory = "1Gi" }
    normalize = { cpu = "1", memory = "1Gi" }
    render    = { cpu = "2", memory = "2Gi" }
  }
}

# The worker SA (shared with download) can read/write the processed bucket, where
# every preprocessing stage writes its output (convert/normalize PLYs, render PNGs).
resource "google_storage_bucket_iam_member" "worker_processed" {
  bucket = google_storage_bucket.processed.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.worker.email}"
}

# Each stage publishes the next stage's job (download→convert→normalize→render), so
# the worker SA must publish to each stage topic.
resource "google_pubsub_topic_iam_member" "worker_publisher" {
  for_each = local.preprocessing_stages
  topic    = google_pubsub_topic.stage[each.key].name
  role     = "roles/pubsub.publisher"
  member   = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_pubsub_topic" "stage" {
  for_each   = local.preprocessing_stages
  name       = "${each.key}-jobs"
  depends_on = [google_project_service.enabled]
}

resource "google_pubsub_topic" "stage_dlq" {
  for_each   = local.preprocessing_stages
  name       = "${each.key}-jobs-dlq"
  depends_on = [google_project_service.enabled]
}

resource "google_cloud_run_v2_service" "stage" {
  for_each            = local.preprocessing_stages
  name                = "${each.key}-worker"
  location            = var.region
  deletion_protection = false

  template {
    service_account = google_service_account.worker.email
    timeout         = "600s" # a single model's preprocessing can take a while

    scaling {
      min_instance_count = 0 # scale to zero
      max_instance_count = 5
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/imagegenie/worker:latest"
      # Override the image's default (local pull worker) with the push receiver.
      # $$ escapes so the shell (not Terraform) expands Cloud Run's $PORT.
      command = ["sh", "-c", "uvicorn app.web:app --host 0.0.0.0 --port $${PORT:-8080}"]

      resources {
        limits = {
          cpu    = local.stage_resources[each.key].cpu
          memory = local.stage_resources[each.key].memory
        }
      }

      env {
        name  = "IMAGEGENIE_STAGE"
        value = each.key
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

resource "google_pubsub_subscription" "stage_worker" {
  for_each = local.preprocessing_stages
  name     = "${each.key}-worker"
  topic    = google_pubsub_topic.stage[each.key].id

  ack_deadline_seconds = 600

  # Push delivery to the stage's Cloud Run service, authenticated by the push SA's
  # OIDC token (server.md#compute); 2xx acks, 5xx nacks.
  push_config {
    push_endpoint = "${google_cloud_run_v2_service.stage[each.key].uri}/pubsub/push"
    oidc_token {
      service_account_email = google_service_account.pubsub_push.email
      audience              = google_cloud_run_v2_service.stage[each.key].uri
    }
  }

  # After 5 failed deliveries, quarantine the message instead of looping forever.
  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.stage_dlq[each.key].id
    max_delivery_attempts = 5
  }

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }
}

# Keep dead-lettered messages inspectable instead of dropping them.
resource "google_pubsub_subscription" "stage_dlq" {
  for_each = local.preprocessing_stages
  name     = "${each.key}-jobs-dlq-sub"
  topic    = google_pubsub_topic.stage_dlq[each.key].id
}

# Dead-lettering needs the Pub/Sub service agent to publish to the DLQ topic and ack
# on the source subscription (local.pubsub_agent is defined in pubsub.tf).
resource "google_pubsub_topic_iam_member" "stage_dlq_publisher" {
  for_each = local.preprocessing_stages
  topic    = google_pubsub_topic.stage_dlq[each.key].name
  role     = "roles/pubsub.publisher"
  member   = local.pubsub_agent
}

resource "google_pubsub_subscription_iam_member" "stage_subscriber" {
  for_each     = local.preprocessing_stages
  subscription = google_pubsub_subscription.stage_worker[each.key].name
  role         = "roles/pubsub.subscriber"
  member       = local.pubsub_agent
}

# The push SA may invoke each (private) stage service. Its token-creator grant on
# the push SA is shared and already made in pubsub.tf.
resource "google_cloud_run_v2_service_iam_member" "stage_push_invoker" {
  for_each = local.preprocessing_stages
  name     = google_cloud_run_v2_service.stage[each.key].name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.pubsub_push.email}"
}
