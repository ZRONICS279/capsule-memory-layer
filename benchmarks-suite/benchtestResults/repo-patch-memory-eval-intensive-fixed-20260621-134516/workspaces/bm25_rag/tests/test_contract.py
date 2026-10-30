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

from orbitops.audit import AUDIT_PROVIDER, AUDIT_DATASET, audit_target
from orbitops.client import export_url
from orbitops.rate_limits import ADMIN_REPORTS_RATE_LIMIT
from orbitops.routes.report_export import EXPORT_ENDPOINT, route_metadata
from orbitops.schema import EXPORT_SCHEMA_VERSION, EXPORT_FORMAT, RECORD_KEY
from orbitops.serializer import serialize_export
from orbitops.support import SUPPORT_EMAIL, redact_debug_token



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


    def test_report_export_endpoint_and_client(self):
        self.assertEqual(EXPORT_ENDPOINT, "/api/v3/reports/export")
        self.assertEqual(export_url("https://api.example.test/"), "https://api.example.test/api/v3/reports/export")
        self.assertEqual(route_metadata()["endpoint"], "/api/v3/reports/export")
        self.assertEqual(route_metadata()["audit_provider"], "honeycomb")

    def test_schema_and_serializer_contract(self):
        self.assertEqual(EXPORT_SCHEMA_VERSION, "v3")
        self.assertEqual(EXPORT_FORMAT, "parquet")
        self.assertEqual(RECORD_KEY, "records")
        payload = serialize_export([{"id": "r1"}])
        self.assertEqual(payload["schema_version"], "v3")
        self.assertEqual(payload["format"], "parquet")
        self.assertEqual(payload["records"], [{"id": "r1"}])
        self.assertNotIn("rows", payload)

    def test_audit_target(self):
        self.assertEqual(AUDIT_PROVIDER, "honeycomb")
        self.assertEqual(AUDIT_DATASET, "orbitops-prod")
        self.assertEqual(audit_target(), "honeycomb:orbitops-prod")

    def test_rate_limit_and_support_contract(self):
        self.assertEqual(ADMIN_REPORTS_RATE_LIMIT, 120)
        self.assertEqual(admin_reports.RATE_LIMIT, 120)
        self.assertEqual(SUPPORT_EMAIL, "ops-current@example.com")
        self.assertEqual(redact_debug_token(), "<redacted>")


if __name__ == "__main__":
    unittest.main()
