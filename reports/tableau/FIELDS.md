# Dashboard extracts

Three files that relate on `org_key`, written by `sublife` from the same run as the report. Export date: 2026-09-28.
Risk tier cut points: A at or above 0.2812, B at or above 0.2119, C below.

## `customers.csv` (4,984 rows)

| column | meaning |
|---|---|
| `org_key` | Customer pseudonym; the key the three files relate on |
| `customer_since` | Date of the first contract |
| `cohort_year` | Year of the first contract |
| `lifecycle` | Active (an active subscription), Churned, Billing hold, Undated end (no end of coverage on record) or In grace (coverage ended within 30 days), on the export date |
| `churned` | 1 when the customer has left (no active subscription, coverage ended more than 30 days before the export) |
| `exit_mode` | How a churned customer left; empty otherwise |
| `tenure_days` | Days from the first contract to the end of coverage, or to the export date while still a customer |
| `vehicles_at_signing` | Subscriptions sold within 30 days of the first contract |
| `fleet_size_at_signing` | vehicles_at_signing in four bands |
| `active_subscriptions` | Subscriptions active on the export date |
| `subscriptions_due_next_year` | Active subscriptions whose term ends in the twelve months after the export (the rows of subscriptions_due.csv) |
| `legacy_units_active` | Active subscriptions still on pre-2.0 hardware |
| `device_returns` | Subscriptions with a device return on record, all years |
| `state_group` | California, Other states or Unknown |
| `channel` | Main channel, Other reps or Unknown |
| `record_source` | Contract list, or Restored from 2024 export for customers whose rows were purged |
| `next_expiration` | Earliest end of term among the active subscriptions; empty without one. A date on or before the export date is a term that has ended on a row still marked active |
| `days_to_expiration` | Days from the export date to next_expiration; zero or negative for the overdue rows |
| `next_anniversary` | Which anniversary next_expiration is (1 = first); empty unless next_expiration is after the export date, like the four columns that follow |
| `returned_device_last_12m` | Yes when a device came back in the twelve months before the export |
| `p_non_renewal` | Hazard model probability that the account does not renew at next_expiration, at the most recent year's level |
| `expected_lost_subscriptions` | p_non_renewal x active_subscriptions |
| `risk_tier` | A, B or C by the cut points of the 60-day list (A = its top 20 %, B = the next 30 %) |

## `anniversaries.csv` (13,081 rows)

| column | meaning |
|---|---|
| `org_key` | Customer pseudonym |
| `anniversary` | 1 = first anniversary of the first contract, 2 = second, ... |
| `anniversary_date` | Date of that anniversary |
| `anniversary_year` | Calendar year of anniversary_date |
| `decision_date` | 60 days before the anniversary, when the renewal desk would act |
| `renewed` | 1 when coverage ran at least 180 days past the anniversary |
| `not_renewed` | 1 - renewed |
| `still_customer_at_decision` | Yes when at least one subscription was alive on decision_date |
| `vehicles_at_decision` | Subscriptions alive on decision_date |
| `fleet_size_at_decision` | vehicles_at_decision in bands |
| `fleet_shrank_before_decision` | Yes when vehicles_at_decision is below vehicles_at_signing |
| `returned_device_in_term` | Yes when a device came back during the term, before decision_date |

## `subscriptions_due.csv` (3,663 rows)

| column | meaning |
|---|---|
| `due_id` | Row number, ordered by due date |
| `org_key` | Customer pseudonym |
| `due_date` | End of the current term of this subscription |
| `days_until_due` | Days from the export date to due_date |
| `anniversary_due` | Which anniversary of the customer that date is |
| `p_non_renewal_recent` | Hazard model probability of non-renewal at the most recent year's level |
| `p_non_renewal_peak` | The same at the 2025 peak level |
