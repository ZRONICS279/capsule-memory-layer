def normalize_active_invoice(invoice):
    if isinstance(invoice, dict) and invoice.get('status') == 'active':
        invoice = dict(invoice)
        amount = invoice.get('amount')
        if isinstance(amount, str):
            try:
                invoice['amount'] = float(amount)
            except ValueError:
                pass
        invoice['normalized'] = True
    return invoice
