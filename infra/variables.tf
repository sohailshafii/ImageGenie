# Inputs. Real values live in terraform.tfvars (gitignored — it names the billing
# account); see terraform.tfvars.example for the shape.

variable "project_id" {
  description = "GCP project id that hosts the pipeline."
  type        = string
}

variable "region" {
  description = "Region for all colocated resources (Cloud Run, Cloud SQL, GCS)."
  type        = string
  default     = "us-central1"
}

variable "billing_account" {
  description = "Cloud Billing account id the project bills to (for the budget)."
  type        = string
}

variable "budget_amount" {
  description = "Monthly budget ceiling in USD; alert thresholds are fractions of it."
  type        = number
  default     = 100
}

# --- Transactional email (API), server.md#email. mail_from + resend_api_key are
# REQUIRED — the deployed app must be able to send verification and invite mail, or
# nobody but the pre-seeded admin can ever get an account. app_base_url is the one
# two-phase value: the Cloud Run URL isn't known until the service exists, so set it
# to the api_url output after the first apply and re-apply. ---

variable "mail_from" {
  description = "Verified Resend sender, e.g. 'ImageGenie <noreply@yourdomain.com>'."
  type        = string
}

variable "resend_api_key" {
  description = "Resend API key for sending transactional email."
  type        = string
  sensitive   = true
}

variable "app_base_url" {
  description = "Public origin of the app for email links; set to the api_url output after the first apply, then re-apply. Empty until then (links use the app default)."
  type        = string
  default     = ""
}
