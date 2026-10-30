from orbitops.auth import LegacyApiKeyAuth

AUTH_MIDDLEWARE = LegacyApiKeyAuth
RATE_LIMIT = 60


def can_access_admin_reports(request):
    return AUTH_MIDDLEWARE()(request)
