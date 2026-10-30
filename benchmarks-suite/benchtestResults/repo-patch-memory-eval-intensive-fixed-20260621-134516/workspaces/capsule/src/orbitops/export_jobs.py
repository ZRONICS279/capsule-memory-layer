PAGINATION_MODE = "cursor"
TIMEOUT_SECONDS = 30

EXPORT_SCHEMA_VERSION = "v3"
EXPORT_FORMAT = "parquet"
RECORD_KEY = "records"


def export_config():
    return {
        "pagination": PAGINATION_MODE,
        "timeout": TIMEOUT_SECONDS,
    }
