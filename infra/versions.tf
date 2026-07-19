# Terraform + provider pinning and the Google provider config. All resources are
# created in var.project_id / var.region (server.md#cloud-platform-gcp).

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
