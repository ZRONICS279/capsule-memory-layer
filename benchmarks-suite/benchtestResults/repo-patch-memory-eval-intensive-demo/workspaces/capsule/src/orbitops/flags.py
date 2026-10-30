CHECKOUT_V2 = 'enabled'
PAYMENT_PLATFORM_OWNER = '@carol-wu'


def enabled_flags():
    return [name for name, value in {"CHECKOUT_V2": CHECKOUT_V2}.items() if value == "enabled"]
