# Monthly budget + alert thresholds (cost guardrail, CLAUDE.md#cost-guardrails).
# Alerts email the billing account's admins/users when this project's spend
# crosses 25% / 50% / 75% / 100% of budget_amount. Creating a budget is free.

data "google_project" "this" {
  project_id = var.project_id
}

resource "google_billing_budget" "monthly" {
  billing_account = var.billing_account
  display_name    = "ImageGenie monthly budget"

  # Scope the budget to this project's spend only.
  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.budget_amount)
    }
  }

  # $25 / $50 / $75 / $100 of the $100 ceiling (CLAUDE.md billing alerts).
  dynamic "threshold_rules" {
    for_each = [0.25, 0.5, 0.75, 1.0]
    content {
      threshold_percent = threshold_rules.value
      spend_basis       = "CURRENT_SPEND"
    }
  }

  depends_on = [google_project_service.enabled]
}
