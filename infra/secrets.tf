# The full DB connection URL (with password) as a Secret Manager secret, injected
# into the Cloud Run worker so the password never sits in plain env/config. Uses
# the Cloud SQL unix socket the connector mounts at /cloudsql/<connection_name>.

locals {
  database_url = "postgresql+psycopg://${google_sql_user.app.name}:${random_password.db.result}@/${google_sql_database.app.name}?host=/cloudsql/${google_sql_database_instance.main.connection_name}"
}

resource "google_secret_manager_secret" "database_url" {
  secret_id = "imagegenie-database-url"

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled]
}

resource "google_secret_manager_secret_version" "database_url" {
  secret      = google_secret_manager_secret.database_url.id
  secret_data = local.database_url
}
