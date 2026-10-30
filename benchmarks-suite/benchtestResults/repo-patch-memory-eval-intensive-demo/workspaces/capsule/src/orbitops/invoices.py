from orbitops.audit import AUDIT_PROVIDER, AUDIT_DATASET
from orbitops.rate_limits import ADMIN_REPORTS_RATE_LIMIT
from orbitops.routes.report_export import EXPORT_ENDPOINT
from orbitops.schema import EXPORT_SCHEMA_VERSION
from orbitops.support import SUPPORT_EMAIL
from orbitops.flags import CHECKOUT_V2
from orbitops.oncall import PAYMENT_PLATFORM_OWNER

DATASET = AUDIT_DATASET
SUPPORT_EMAIL = SUPPORT_EMAIL
ADMIN_REPORTS_RATE_LIMIT = ADMIN_REPORTS_RATE_LIMIT
EXPORT_ENDPOINT = EXPORT_ENDPOINT
AUDIT_PROVIDER = AUDIT_PROVIDER
EXPORT_SCHEMA_VERSION = EXPORT_SCHEMA_VERSION
CHECKOUT_V2 = CHECKOUT_V2
PAYMENT_PLATFORM_OWNER = PAYMENT_PLATFORM_OWNER


def normalize_active_invoice(invoice):
    normalized = dict(invoice)
    amount = normalized.get("amount")
    if isinstance(amount, str):
        try:
            normalized["amount"] = float(amount)
        except ValueError:
            pass
    normalized.setdefault("currency", "USD")
    normalized["status"] = "active"
    return normalized
