from orbitops.auth import LegacyApiKeyAuth

AUTH_MIDDLEWARE = LegacyApiKeyAuth


def can_access_admin_reports(request):
    return AUTH_MIDDLEWARE()(request)
