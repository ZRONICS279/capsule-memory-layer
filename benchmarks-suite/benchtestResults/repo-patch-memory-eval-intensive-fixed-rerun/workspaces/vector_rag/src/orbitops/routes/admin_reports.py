from orbitops.auth import RequireSessionToken

AUTH_MIDDLEWARE = RequireSessionToken
RATE_LIMIT = 120


def can_access_admin_reports(request):
    return AUTH_MIDDLEWARE()(request)
