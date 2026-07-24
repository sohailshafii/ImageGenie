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

# --- Optional transactional email (API). All empty ⇒ email disabled: the app logs
# verification/invite links instead of sending them, which is fine until signup for
# users other than the pre-seeded admin is needed (server.md#email). ---

variable "mail_from" {
  description = "Verified Resend sender, e.g. 'ImageGenie <noreply@...>'. Empty disables email."
  type        = string
  default     = ""
}

variable "app_base_url" {
  description = "Public origin of the app for email links (set to the api_url output after first apply). Empty disables email links."
  type        = string
  default     = ""
}

variable "resend_api_key" {
  description = "Resend API key for sending email. Empty disables sending."
  type        = string
  default     = ""
  sensitive   = true
}
