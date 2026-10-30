from orbitops.auth import RequireSessionToken
from orbitops.rate_limits import ADMIN_REPORTS_RATE_LIMIT

AUTH_MIDDLEWARE = RequireSessionToken
RATE_LIMIT = ADMIN_REPORTS_RATE_LIMIT


def report_route_config():
    return {
        "auth_middleware": AUTH_MIDDLEWARE,
        "rate_limit": RATE_LIMIT,
    }
