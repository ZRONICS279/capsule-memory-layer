PAGINATION_MODE = "cursor"
TIMEOUT_SECONDS = 30


def export_config():
    return {
        "pagination": PAGINATION_MODE,
        "timeout": TIMEOUT_SECONDS,
    }
