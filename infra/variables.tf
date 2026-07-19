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
