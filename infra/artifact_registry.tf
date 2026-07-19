# Artifact Registry Docker repo for the pipeline's container images. Creating the
# repo is free; stored images cost ~$0.10/GB/mo (a slim worker image is well under
# a dollar). Cloud Run pulls the worker image from here.

resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = "imagegenie"
  format        = "DOCKER"
  description   = "Worker / API container images for the pipeline."

  depends_on = [google_project_service.enabled]
}
