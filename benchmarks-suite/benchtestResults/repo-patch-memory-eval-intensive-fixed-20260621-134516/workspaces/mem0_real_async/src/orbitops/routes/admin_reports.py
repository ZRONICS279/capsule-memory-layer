from orbitops.auth import RequireSessionToken
from orbitops.rate_limits import ADMIN_REPORTS_RATE_LIMIT

AUTH_MIDDLEWARE = RequireSessionToken
RATE_LIMIT = ADMIN_REPORTS_RATE_LIMIT


def can_access_admin_reports(request):
    return AUTH_MIDDLEWARE()(request)
