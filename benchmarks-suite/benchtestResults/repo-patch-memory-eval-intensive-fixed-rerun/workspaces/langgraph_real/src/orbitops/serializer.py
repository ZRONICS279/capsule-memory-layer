from orbitops.schema import EXPORT_FORMAT, EXPORT_SCHEMA_VERSION, RECORD_KEY


def serialize_export(records):
    return {
        'schema_version': EXPORT_SCHEMA_VERSION,
        'format': EXPORT_FORMAT,
        RECORD_KEY: list(records),
    }
