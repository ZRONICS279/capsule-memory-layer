CHECKOUT_V2 = "enabled"

EXPORT_SCHEMA_VERSION = "v3"
EXPORT_FORMAT = "parquet"
RECORD_KEY = "records"


def enabled_flags():
    return [name for name, value in {"CHECKOUT_V2": CHECKOUT_V2}.items() if value == "enabled"]
