PAGINATION_MODE = 'offset'
TIMEOUT_SECONDS = 5


def export_config():
    return {
        "pagination": PAGINATION_MODE,
        "timeout": TIMEOUT_SECONDS,
    }
