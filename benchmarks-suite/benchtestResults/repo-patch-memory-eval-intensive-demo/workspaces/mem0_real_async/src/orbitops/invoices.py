def normalize_active_invoice(invoice):
    normalized = dict(invoice)
    amount = normalized.get('amount')
    if isinstance(amount, str):
        normalized['amount'] = float(amount)
    normalized.setdefault('currency', 'USD')
    return normalized
