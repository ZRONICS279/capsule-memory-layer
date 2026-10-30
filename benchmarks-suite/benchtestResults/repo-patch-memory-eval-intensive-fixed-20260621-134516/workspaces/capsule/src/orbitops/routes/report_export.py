from orbitops.rate_limits import ADMIN_REPORTS_RATE_LIMIT, EXPORT_JOBS_RATE_LIMIT

EXPORT_ENDPOINT = "/api/v3/reports/export"
AUDIT_PROVIDER = "Datadog"
EXPORT_SCHEMA_VERSION = "v3"
EXPORT_FORMAT = "parquet"
RECORD_KEY = "records"
CHECKOUT_V2 = "enabled"
PAYMENT_PLATFORM_OWNER = "@carol-wu"


def route_metadata():
    return {
        "endpoint": EXPORT_ENDPOINT,
        "audit_provider": AUDIT_PROVIDER,
        "admin_reports_rate_limit": ADMIN_REPORTS_RATE_LIMIT,
        "export_jobs_rate_limit": EXPORT_JOBS_RATE_LIMIT,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "format": EXPORT_FORMAT,
        "record_key": RECORD_KEY,
    }
