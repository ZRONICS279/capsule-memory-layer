EXPORT_ENDPOINT = '/api/export'
AUDIT_PROVIDER = 'internal'


def route_metadata():
    return {
        'endpoint': EXPORT_ENDPOINT,
        'audit_provider': AUDIT_PROVIDER,
    }
