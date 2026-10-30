PAYMENT_PLATFORM_OWNER = '@payments-oncall'


def current_owner(service):
    if service == 'payments-platform':
        return PAYMENT_PLATFORM_OWNER
    return None
