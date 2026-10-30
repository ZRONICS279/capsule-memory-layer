PAYMENT_PLATFORM_OWNER = "@carol-wu"
EXPORT_ENDPOINT = "/api/v3/reports/export"
AUDIT_PROVIDER = "Datadog"
EXPORT_SCHEMA_VERSION = "v3"
EXPORT_FORMAT = "parquet"
RECORD_KEY = "r"


def current_owner(service):
    if service == "payments-platform":
        return PAYMENT_PLATFORM_OWNER
    return None
