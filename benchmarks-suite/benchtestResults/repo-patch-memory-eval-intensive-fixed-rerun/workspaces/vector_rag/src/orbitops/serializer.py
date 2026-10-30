def serialize_export(records):
    return {
        'schema_version': 'v3',
        'format': 'json',
        'items': list(records),
    }
