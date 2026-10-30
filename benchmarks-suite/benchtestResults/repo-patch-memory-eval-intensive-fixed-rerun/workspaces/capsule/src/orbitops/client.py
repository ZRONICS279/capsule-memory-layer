EXPORT_ENDPOINT = '/api/v2/export'


def export_url(base_url):
    return base_url.rstrip("/") + EXPORT_ENDPOINT
