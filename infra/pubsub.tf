# Pub/Sub queue (server.md#queue): one topic + subscription per stage, each with
# its own dead-letter topic. Milestone 3 provisions the download stage — the only
# worker built so far; convert/normalize/render topics land with those workers.
# Names match server/app/config.py so the deployed worker connects with no extra
# config. Topics/subscriptions cost nothing at idle.

resource "google_pubsub_topic" "download" {
  name       = "download-jobs"
  depends_on = [google_project_service.enabled]
}

# Quarantine for "poison" messages that fail every delivery attempt.
resource "google_pubsub_topic" "download_dlq" {
  name       = "download-jobs-dlq"
  depends_on = [google_project_service.enabled]
}

resource "google_pubsub_subscription" "download_worker" {
  name  = "download-worker"
  topic = google_pubsub_topic.download.id

  ack_deadline_seconds = 600 # a single model download can take a while

  # Push delivery: Pub/Sub POSTs each message to the Cloud Run service, minting an
  # OIDC token as the push SA so Cloud Run's IAM authenticates the call. The 2xx/5xx
  # response is the ack/nack (server.md#compute).
  push_config {
    push_endpoint = "${google_cloud_run_v2_service.download.uri}/pubsub/push"
    oidc_token {
      service_account_email = google_service_account.pubsub_push.email
      audience              = google_cloud_run_v2_service.download.uri
    }
  }

  # After 5 failed deliveries, route the message to the dead-letter topic instead
  # of redelivering forever (server.md#queue).
  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.download_dlq.id
    max_delivery_attempts = 5
  }

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }
}

# Keep failed messages inspectable instead of dropping them on the floor.
resource "google_pubsub_subscription" "download_dlq" {
  name  = "download-jobs-dlq-sub"
  topic = google_pubsub_topic.download_dlq.id
}

# Dead-lettering requires the Pub/Sub service agent to publish to the DLQ topic and
# to ack messages on the source subscription. Grant exactly those.
locals {
  pubsub_agent = "serviceAccount:service-${data.google_project.this.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

resource "google_pubsub_topic_iam_member" "dlq_publisher" {
  topic  = google_pubsub_topic.download_dlq.name
  role   = "roles/pubsub.publisher"
  member = local.pubsub_agent
}

resource "google_pubsub_subscription_iam_member" "worker_subscriber" {
  subscription = google_pubsub_subscription.download_worker.name
  role         = "roles/pubsub.subscriber"
  member       = local.pubsub_agent
}

# --- Push auth: the identity Pub/Sub uses to invoke the Cloud Run push endpoint ---
resource "google_service_account" "pubsub_push" {
  account_id   = "pubsub-push"
  display_name = "Pub/Sub push invoker"
}

# The push SA may invoke the (private) Cloud Run service.
resource "google_cloud_run_v2_service_iam_member" "push_invoker" {
  name     = google_cloud_run_v2_service.download.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.pubsub_push.email}"
}

# Pub/Sub's own service agent must be able to mint OIDC tokens as the push SA.
resource "google_service_account_iam_member" "pubsub_token_creator" {
  service_account_id = google_service_account.pubsub_push.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = local.pubsub_agent
}
