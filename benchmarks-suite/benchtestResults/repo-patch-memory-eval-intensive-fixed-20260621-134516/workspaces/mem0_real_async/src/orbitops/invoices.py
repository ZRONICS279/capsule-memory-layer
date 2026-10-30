def normalize_legacy_invoice(invoice):
    return {
        "amount": invoice.get("amount"),
        "currency": invoice.get("currency"),
        "status": invoice.get("status"),
    }
