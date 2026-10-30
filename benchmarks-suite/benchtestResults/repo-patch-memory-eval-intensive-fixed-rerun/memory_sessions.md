## Session 1

- Session 1 auth brainstorm: admin reports initially used LegacyApiKeyAuth for quick testing.
- DECISION SUPERSEDED: AUTH_MIDDLEWARE = LegacyApiKeyAuth in src/orbitops/routes/admin_reports.py.
- Security notes: API keys are simple but do not validate expiry or user sessions.

## Session 2

- Session 2 security review: LegacyApiKeyAuth is rejected for admin reports.
- FINAL AUTH DECISION: define RequireSessionToken in src/orbitops/auth.py.
- FINAL AUTH DECISION: src/orbitops/routes/admin_reports.py must import RequireSessionToken and set AUTH_MIDDLEWARE = RequireSessionToken.
- Do not leave LegacyApiKeyAuth anywhere in active code.

## Session 3

- Session 3 export jobs performance: offset pagination caused memory spikes on large exports.
- Session 3 export timeout: TIMEOUT_SECONDS=5 is too low for 10k-row exports.
- FINAL EXPORT CONFIG: PAGINATION_MODE='cursor' and TIMEOUT_SECONDS=30 in src/orbitops/export_jobs.py.
- Rejected export config: PAGINATION_MODE='offset' and TIMEOUT_SECONDS=5.

## Session 4

- Session 4 invoice docs are stale. The docs mention normalize_legacy_invoice, but that function is rejected.
- FINAL INVOICE DECISION: expose normalize_active_invoice in src/orbitops/invoices.py.
- normalize_active_invoice should return id, amount as float, currency default USD, and status active.
- Do not keep normalize_legacy_invoice in active code.

## Session 5

- Session 5 checkout rollout: CHECKOUT_V2 was disabled, then canary, then disabled after a Safari incident.
- FINAL CHECKOUT DECISION: Safari fix shipped. Set CHECKOUT_V2='enabled' in src/orbitops/flags.py.
- Rejected checkout states: 'disabled' and 'canary'.

## Session 6

- Session 6 ownership history: @alice-chen owned payments-platform, then @bob-martinez, then @platform-team interim.
- FINAL OWNER DECISION: payments-platform owner is @carol-wu.
- Update src/orbitops/oncall.py so PAYMENT_PLATFORM_OWNER='@carol-wu'.
- Rejected owners: @alice-chen, @bob-martinez, @platform-team.

## Session 7

- Session 7 dependency review: pydantic==1.10.0 and pydantic==2.6.4 were both rejected during compatibility testing.
- FINAL DEPENDENCY DECISION: pin pydantic==2.8.2 in requirements.txt.
- Do not leave the rejected pydantic pins in requirements.txt.

## Session 8

- Session 8 API contract history: /api/v1/export was deprecated, /api/v2/export was trialed, and both are rejected.
- FINAL API CONTRACT: EXPORT_ENDPOINT='/api/v3/reports/export' in src/orbitops/routes/report_export.py.
- src/orbitops/client.py must call /api/v3/reports/export, not /api/v1/export or /api/v2/export.

## Session 9

- Session 9 schema migration: schema_version=v1 and schema_version=v2 were both migration-only drafts.
- FINAL SCHEMA DECISION: EXPORT_SCHEMA_VERSION='v3' and EXPORT_FORMAT='parquet' in src/orbitops/schema.py.
- src/orbitops/serializer.py must emit schema_version v3, format parquet, and use records instead of rows.

## Session 10

- Session 10 audit provider review: Datadog and Grafana Cloud were rejected for the export audit trail.
- FINAL AUDIT DECISION: AUDIT_PROVIDER='honeycomb' and AUDIT_DATASET='orbitops-prod' in src/orbitops/audit.py.
- src/orbitops/routes/report_export.py should expose audit provider honeycomb in route_metadata().

## Session 11

- Session 11 rate-limit review: admin reports rate limit 60/min and 90/min were too low after load testing.
- FINAL RATE LIMIT DECISION: ADMIN_REPORTS_RATE_LIMIT=120 in src/orbitops/rate_limits.py.
- src/orbitops/routes/admin_reports.py must expose RATE_LIMIT = ADMIN_REPORTS_RATE_LIMIT.

## Session 12

- Session 12 deletion/privacy note: support email ops-old@example.com and debug token debug-secret-123 were pasted in old runbooks.
- FINAL PRIVACY DECISION: active support email is ops-current@example.com. Do not keep debug-secret-123 in active code.
- Update src/orbitops/support.py with SUPPORT_EMAIL='ops-current@example.com' and redact_debug_token() returning '<redacted>'.