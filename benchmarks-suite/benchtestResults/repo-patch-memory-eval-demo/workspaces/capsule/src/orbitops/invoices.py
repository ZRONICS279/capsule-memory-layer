def normalize_active_invoice(invoice):
    normalized = dict(invoice)
    amount = normalized.get("amount")
    if amount is not None:
        normalized["amount"] = float(amount)
    normalized.setdefault("currency", "USD")
    normalized.setdefault("status", "active")
    return normalized
