def normalize_active_invoice(invoice):
    normalized = dict(invoice)
    normalized["amount"] = float(normalized["amount"])
    normalized.setdefault("currency", "USD")
    normalized.setdefault("status", "active")
    return normalized
