EXPORT_ENDPOINT = '/api/export'


def export_url(base_url):
    return base_url.rstrip('/') + EXPORT_ENDPOINT
