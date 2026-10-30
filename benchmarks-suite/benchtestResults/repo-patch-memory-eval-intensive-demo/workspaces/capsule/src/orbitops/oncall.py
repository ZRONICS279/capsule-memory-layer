PAYMENT_PLATFORM_OWNER = '@carol-wu'
CHECKOUT_V2 = 'enabled'


def current_owner(service):
    if service == "payments-platform":
        return PAYMENT_PLATFORM_OWNER
    return None
