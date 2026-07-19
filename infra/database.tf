# Cloud SQL for PostgreSQL (server.md#database) — the pipeline's metadata source of
# truth. Smallest shared-core tier, single zone, no HA.
#
# COST NOTE: unlike the scale-to-zero services, Cloud SQL is ALWAYS ON, so it's the
# main recurring charge (~$8-10/mo for db-f1-micro + 10GB). `terraform destroy`
# stops the billing.

resource "random_password" "db" {
  length  = 24
  special = false
}

resource "google_sql_database_instance" "main" {
  name             = "imagegenie-pg"
  database_version = "POSTGRES_16"
  region           = var.region

  settings {
    tier              = "db-f1-micro" # smallest shared-core
    edition           = "ENTERPRISE"
    availability_type = "ZONAL"  # single zone (no HA) — cheapest
    disk_size         = 10       # GB (minimum)
    disk_type         = "PD_HDD" # a metadata DB is light; HDD is cheaper
    disk_autoresize   = false    # cap growth to bound cost

    backup_configuration {
      enabled = false # metadata is reproducible by re-running the pipeline
    }

    ip_configuration {
      ipv4_enabled = true # public IP; Cloud Run connects via the Cloud SQL connector
    }
  }

  deletion_protection = false # dev project: allow terraform destroy

  depends_on = [google_project_service.enabled]
}

resource "google_sql_database" "app" {
  name     = "imagegenie"
  instance = google_sql_database_instance.main.name
}

resource "google_sql_user" "app" {
  name     = "imagegenie"
  instance = google_sql_database_instance.main.name
  password = random_password.db.result
}
