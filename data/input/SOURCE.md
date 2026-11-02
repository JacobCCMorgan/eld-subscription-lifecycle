# Input data: provenance and de-identification

All tables are de-identified extracts of CRM and operations data from an ELD / fleet-telematics vendor
(about 5,300 customer organizations, 2019-2026). Every identifier is a keyed pseudonym, no names, contact
details, free text or payment fields are present, and the key is not in this repository. The same key was
used for the [`telematics-data-foundation`](https://github.com/JacobCCMorgan/telematics-data-foundation)
tables, so `org_key`, `dot_key`, `device_key` and `vin_key` join across repositories.

| file | rows | grain | origin |
|---|---:|---|---|
| `fact_contract.parquet` | 13,759 | one row per vehicle subscription slot as of 28 Sep 2026 | CRM contract list; copied unchanged from the foundation public tables (commit `22dada6`) |
| `dim_org.parquet` | 5,675 | one row per organization | CRM organization list joined with the app's company master; from the foundation |
| `dim_device.parquet` | 4,970 | one row per ELD unit (device registry snapshot, 28 Sep 2026) | from the foundation; used for the device-generation map |
| `contracts_snapshot_2024-01-05.parquet` | 4,977 | one row per *active* vehicle subscription as of 5 Jan 2024 | an earlier export of the same CRM list, kept in an operations workbook; pseudonymized with the same key |
| `migration_calls.parquet` | 9,509 | one row per organization-vehicle listed in a call round | the ELD 2.0 migration call campaign: five call rounds (three in 2025, two in September 2026); only the call result category, round, date (when recorded), staff and sales-rep pseudonyms and two fleet flags are kept |
| `migration_pending_2026-01-15.parquet` | 1,374 | one row per vehicle not yet migrated on 15 Jan 2026 | the operations team's pending list; organization, status and sales-rep pseudonym only |

## What the contract columns mean (important)

The CRM keeps **one row per vehicle slot and overwrites the term dates in place at each renewal**:

- `contract_date` is the date the slot was first sold and is inherited by a replacement device;
- `activation_date` is the start of the **current** one-year term, rolled forward by a year at every
  renewal (93 % of `activation_date - contract_date` gaps are whole multiples of 365 days);
- `expiration_date` is the end of the current term;
- when a device is replaced, a new row is created with `prev_device_key` pointing at the old unit and the
  old row is set to `deleted`, usually with a `return_date`.

So the number of renewals a slot has gone through is `round((activation_date - contract_date) / 365)`,
and the January 2024 snapshot is what the same rows looked like two and a half years earlier. The
analysis code reconstructs renewal histories from these semantics; see `src/sublife/orgs.py`.

## De-identification

- `ORG-`, `DOT-`, `DEV-`, `VIN-`, `REP-`, `STF-` values are HMAC-SHA256 pseudonyms with a secret key.
- Call results were reduced to a fixed set of English category codes; the operators' notes are not
  included. Rounds 1-3 have no per-call date in the source (only that the campaign text was sent on 25 June
  of the 2025 outreach); `date_known` is false for those rows.
- `is_internal_test` marks the vendor's internal test organizations; the analysis excludes them.
- Counts below 5 in published aggregate tables are reported as "<5".

The raw workbooks and the key are not part of the repository; `scripts/prepare_inputs.py` documents how
the inputs were produced.
