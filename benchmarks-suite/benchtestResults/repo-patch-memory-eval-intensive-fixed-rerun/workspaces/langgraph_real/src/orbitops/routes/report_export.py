EXPORT_ENDPOINT = '/api/v3/reports/export'
AUDIT_PROVIDER = 'honeycomb'


def route_metadata():
    return {
        "endpoint": EXPORT_ENDPOINT,
        "audit_provider": AUDIT_PROVIDER,
    }
