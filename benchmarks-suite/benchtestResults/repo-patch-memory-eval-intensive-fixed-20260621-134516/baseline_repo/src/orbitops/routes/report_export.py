EXPORT_ENDPOINT = '/api/v1/export'
AUDIT_PROVIDER = 'Datadog'


def route_metadata():
    return {
        "endpoint": EXPORT_ENDPOINT,
        "audit_provider": AUDIT_PROVIDER,
    }
