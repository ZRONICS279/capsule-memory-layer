import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbitops.auth import RequireSessionToken
from orbitops.routes import admin_reports
from orbitops.export_jobs import PAGINATION_MODE, TIMEOUT_SECONDS, export_config
from orbitops.flags import CHECKOUT_V2, enabled_flags
from orbitops.invoices import normalize_active_invoice
from orbitops.oncall import current_owner


class Request:
    session_token = "token"


class ContractTest(unittest.TestCase):
    def test_admin_reports_use_session_token(self):
        self.assertIs(admin_reports.AUTH_MIDDLEWARE, RequireSessionToken)
        self.assertTrue(admin_reports.can_access_admin_reports(Request()))

    def test_export_jobs_final_config(self):
        self.assertEqual(PAGINATION_MODE, "cursor")
        self.assertEqual(TIMEOUT_SECONDS, 30)
        self.assertEqual(export_config(), {"pagination": "cursor", "timeout": 30})

    def test_invoice_normalizer(self):
        result = normalize_active_invoice({"id": "inv_1", "amount": "42.50"})
        self.assertEqual(result["id"], "inv_1")
        self.assertEqual(result["amount"], 42.5)
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["status"], "active")

    def test_checkout_flag_enabled(self):
        self.assertEqual(CHECKOUT_V2, "enabled")
        self.assertEqual(enabled_flags(), ["CHECKOUT_V2"])

    def test_owner_final(self):
        self.assertEqual(current_owner("payments-platform"), "@carol-wu")

    def test_dependency_pin(self):
        requirements = (ROOT / "requirements.txt").read_text()
        self.assertIn("pydantic==2.8.2", requirements)


if __name__ == "__main__":
    unittest.main()
