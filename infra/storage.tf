# GCS buckets (server.md#object-storage): raw meshes vs processed renders, kept
# in separate buckets for independent lifecycles. Colocated in var.region so
# intra-region reads are free (NFR-5). Names are globally unique via the project
# id prefix. force_destroy=true so `terraform destroy` can clear this dev
# project's buckets even when non-empty.

resource "google_storage_bucket" "raw" {
  name                        = "${var.project_id}-raw"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true

  # Storage-class economics (why raw != processed):
  #   Standard  — higher per-GB/month storage price, but NO retrieval fee and NO
  #               minimum storage duration.
  #   Nearline  — ~half the per-GB/month price, BUT a per-GB retrieval fee on every
  #               read AND a 30-day minimum duration (delete sooner and you're still
  #               billed for the full 30 days).
  # Raw meshes are written once and then read ~once (during preprocessing), so the
  # per-read retrieval fee barely applies and they easily outlive 30 days → Nearline
  # is a clear win. Processed renders (below) are read every training epoch, so
  # Nearline's per-read fee would pile up and erase the savings → they stay Standard.
  # The trap avoided: cheap-looking Nearline on hot training data costs *more*.
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  depends_on = [google_project_service.enabled]
}

resource "google_storage_bucket" "processed" {
  # Processed renders stay Standard (training reads them every epoch).
  name                        = "${var.project_id}-processed"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true

  depends_on = [google_project_service.enabled]
}
