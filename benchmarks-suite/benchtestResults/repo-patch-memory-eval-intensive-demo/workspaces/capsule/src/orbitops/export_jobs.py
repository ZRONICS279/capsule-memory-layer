PAGINATION_MODE = 'cursor'
TIMEOUT_SECONDS = 30


def export_job_config():
    return {
        "pagination_mode": PAGINATION_MODE,
        "timeout": TIMEOUT_SECONDS,
    }
