def serialize_export(records):
    return {
        'schema_version': 'v2',
        'format': 'csv',
        'rows': list(records),
    }
