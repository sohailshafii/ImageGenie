# Enable the GCP service APIs the pipeline uses. Enabling is free; resources
# created against them are what cost money. disable_on_destroy = false so a
# `terraform destroy` of resources doesn't also turn off (disruptive) APIs.

locals {
  services = [
    "cloudresourcemanager.googleapis.com", # project/API management (bootstrap)
    "run.googleapis.com",                  # Cloud Run (workers)
    "pubsub.googleapis.com",               # queue
    "sqladmin.googleapis.com",             # Cloud SQL (Postgres)
    "storage.googleapis.com",              # GCS buckets
    "artifactregistry.googleapis.com",     # worker image registry
    "cloudbilling.googleapis.com",         # billing operations
    "billingbudgets.googleapis.com",       # budget + alerts
    "iam.googleapis.com",                  # service accounts / IAM
  ]
}

resource "google_project_service" "enabled" {
  for_each           = toset(local.services)
  service            = each.value
  disable_on_destroy = false
}
