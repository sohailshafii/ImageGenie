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

  # The three.js viewer fetches the normalized PLY cross-origin via a signed GCS
  # URL (PLYLoader uses XHR/fetch), so the bucket must return CORS headers or the
  # browser blocks the read — the mesh silently fails while <img> thumbnails, which
  # are exempt from CORS, still render. Signed URLs make this a production-only bug
  # (local dev streams same-origin through the API). Only GET is needed; the app is
  # the sole origin. Gated on app_base_url so the first apply (URL not yet known)
  # doesn't set an empty origin — CORS lands on the phase-2 re-apply.
  dynamic "cors" {
    for_each = var.app_base_url == "" ? [] : [1]
    content {
      origin          = [var.app_base_url]
      method          = ["GET", "HEAD"]
      response_header = ["Content-Type", "Content-Length", "Content-Range", "Accept-Ranges", "ETag"]
      max_age_seconds = 3600
    }
  }

  depends_on = [google_project_service.enabled]
}
