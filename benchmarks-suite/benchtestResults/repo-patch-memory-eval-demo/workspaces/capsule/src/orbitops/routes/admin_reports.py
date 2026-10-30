from orbitops.auth import RequireSessionToken

AUTH_MIDDLEWARE = RequireSessionToken


def can_access_admin_reports(request):
    return AUTH_MIDDLEWARE()(request)
