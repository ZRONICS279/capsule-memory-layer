from orbitops.audit import AUDIT_PROVIDER

EXPORT_ENDPOINT = '/api/v3/reports/export'


def route_metadata():
    return {
        "endpoint": EXPORT_ENDPOINT,
        "audit_provider": AUDIT_PROVIDER,
    }
