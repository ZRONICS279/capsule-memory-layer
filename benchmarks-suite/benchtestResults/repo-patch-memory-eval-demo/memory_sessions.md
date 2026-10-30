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