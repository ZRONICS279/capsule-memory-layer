PAYMENT_PLATFORM_OWNER = '@alice-chen'


def current_owner(service):
    if service == "payments-platform":
        return PAYMENT_PLATFORM_OWNER
    return None
