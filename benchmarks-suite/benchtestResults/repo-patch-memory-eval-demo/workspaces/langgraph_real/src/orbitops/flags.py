CHECKOUT_V2 = "enabled"


def enabled_flags():
    return [name for name, value in {"CHECKOUT_V2": CHECKOUT_V2}.items() if value == "enabled"]
