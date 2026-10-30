EXPORT_ENDPOINT = '/api/v3/reports/export'


def export_url(base_url):
    return base_url.rstrip("/") + EXPORT_ENDPOINT
