#!/usr/bin/env python3
"""Create identical repo-patch workspaces for memory-layer agent testing."""

from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from datetime import datetime
from pathlib import Path


DEFAULT_ADAPTERS = ("capsule", "mem0_real_async", "langgraph_real", "vector_rag", "bm25_rag")


CORE_MEMORY_SESSIONS = [
    [
        "Session 1 auth brainstorm: admin reports initially used LegacyApiKeyAuth for quick testing.",
        "DECISION SUPERSEDED: AUTH_MIDDLEWARE = LegacyApiKeyAuth in src/orbitops/routes/admin_reports.py.",
        "Security notes: API keys are simple but do not validate expiry or user sessions.",
    ],
    [
        "Session 2 security review: LegacyApiKeyAuth is rejected for admin reports.",
        "FINAL AUTH DECISION: define RequireSessionToken in src/orbitops/auth.py.",
        "FINAL AUTH DECISION: src/orbitops/routes/admin_reports.py must import RequireSessionToken and set AUTH_MIDDLEWARE = RequireSessionToken.",
        "Do not leave LegacyApiKeyAuth anywhere in active code.",
    ],
    [
        "Session 3 export jobs performance: offset pagination caused memory spikes on large exports.",
        "Session 3 export timeout: TIMEOUT_SECONDS=5 is too low for 10k-row exports.",
        "FINAL EXPORT CONFIG: PAGINATION_MODE='cursor' and TIMEOUT_SECONDS=30 in src/orbitops/export_jobs.py.",
        "Rejected export config: PAGINATION_MODE='offset' and TIMEOUT_SECONDS=5.",
    ],
    [
        "Session 4 invoice docs are stale. The docs mention normalize_legacy_invoice, but that function is rejected.",
        "FINAL INVOICE DECISION: expose normalize_active_invoice in src/orbitops/invoices.py.",
        "normalize_active_invoice should return id, amount as float, currency default USD, and status active.",
        "Do not keep normalize_legacy_invoice in active code.",
    ],
    [
        "Session 5 checkout rollout: CHECKOUT_V2 was disabled, then canary, then disabled after a Safari incident.",
        "FINAL CHECKOUT DECISION: Safari fix shipped. Set CHECKOUT_V2='enabled' in src/orbitops/flags.py.",
        "Rejected checkout states: 'disabled' and 'canary'.",
    ],
    [
        "Session 6 ownership history: @alice-chen owned payments-platform, then @bob-martinez, then @platform-team interim.",
        "FINAL OWNER DECISION: payments-platform owner is @carol-wu.",
        "Update src/orbitops/oncall.py so PAYMENT_PLATFORM_OWNER='@carol-wu'.",
        "Rejected owners: @alice-chen, @bob-martinez, @platform-team.",
    ],
    [
        "Session 7 dependency review: pydantic==1.10.0 and pydantic==2.6.4 were both rejected during compatibility testing.",
        "FINAL DEPENDENCY DECISION: pin pydantic==2.8.2 in requirements.txt.",
        "Do not leave the rejected pydantic pins in requirements.txt.",
    ],
]


INTENSIVE_MEMORY_SESSIONS = [
    [
        "Session 8 API contract history: /api/v1/export was deprecated, /api/v2/export was trialed, and both are rejected.",
        "FINAL API CONTRACT: EXPORT_ENDPOINT='/api/v3/reports/export' in src/orbitops/routes/report_export.py.",
        "src/orbitops/client.py must call /api/v3/reports/export, not /api/v1/export or /api/v2/export.",
    ],
    [
        "Session 9 schema migration: schema_version=v1 and schema_version=v2 were both migration-only drafts.",
        "FINAL SCHEMA DECISION: EXPORT_SCHEMA_VERSION='v3' and EXPORT_FORMAT='parquet' in src/orbitops/schema.py.",
        "src/orbitops/serializer.py must emit schema_version v3, format parquet, and use records instead of rows.",
    ],
    [
        "Session 10 audit provider review: Datadog and Grafana Cloud were rejected for the export audit trail.",
        "FINAL AUDIT DECISION: AUDIT_PROVIDER='honeycomb' and AUDIT_DATASET='orbitops-prod' in src/orbitops/audit.py.",
        "src/orbitops/routes/report_export.py should expose audit provider honeycomb in route_metadata().",
    ],
    [
        "Session 11 rate-limit review: admin reports rate limit 60/min and 90/min were too low after load testing.",
        "FINAL RATE LIMIT DECISION: ADMIN_REPORTS_RATE_LIMIT=120 in src/orbitops/rate_limits.py.",
        "src/orbitops/routes/admin_reports.py must expose RATE_LIMIT = ADMIN_REPORTS_RATE_LIMIT.",
    ],
    [
        "Session 12 deletion/privacy note: support email ops-old@example.com and debug token debug-secret-123 were pasted in old runbooks.",
        "FINAL PRIVACY DECISION: active support email is ops-current@example.com. Do not keep debug-secret-123 in active code.",
        "Update src/orbitops/support.py with SUPPORT_EMAIL='ops-current@example.com' and redact_debug_token() returning '<redacted>'.",
    ],
]


CORE_EXPECTED_FILES = [
    "requirements.txt",
    "src/orbitops/auth.py",
    "src/orbitops/routes/admin_reports.py",
    "src/orbitops/export_jobs.py",
    "src/orbitops/invoices.py",
    "src/orbitops/flags.py",
    "src/orbitops/oncall.py",
]


INTENSIVE_EXPECTED_FILES = [
    "src/orbitops/routes/report_export.py",
    "src/orbitops/client.py",
    "src/orbitops/schema.py",
    "src/orbitops/serializer.py",
    "src/orbitops/audit.py",
    "src/orbitops/rate_limits.py",
    "src/orbitops/support.py",
]


CORE_REQUIRED_VALUES = [
    "RequireSessionToken",
    "PAGINATION_MODE = 'cursor'",
    "TIMEOUT_SECONDS = 30",
    "normalize_active_invoice",
    "CHECKOUT_V2 = 'enabled'",
    "PAYMENT_PLATFORM_OWNER = '@carol-wu'",
    "pydantic==2.8.2",
]


INTENSIVE_REQUIRED_VALUES = [
    "EXPORT_ENDPOINT = '/api/v3/reports/export'",
    "EXPORT_SCHEMA_VERSION = 'v3'",
    "EXPORT_FORMAT = 'parquet'",
    "AUDIT_PROVIDER = 'honeycomb'",
    "AUDIT_DATASET = 'orbitops-prod'",
    "ADMIN_REPORTS_RATE_LIMIT = 120",
    "SUPPORT_EMAIL = 'ops-current@example.com'",
]


CORE_FORBIDDEN = {
    "src/orbitops/auth.py": ["LegacyApiKeyAuth"],
    "src/orbitops/routes/admin_reports.py": ["LegacyApiKeyAuth"],
    "src/orbitops/export_jobs.py": ["PAGINATION_MODE = 'offset'", "TIMEOUT_SECONDS = 5"],
    "src/orbitops/invoices.py": ["normalize_legacy_invoice"],
    "src/orbitops/flags.py": ["CHECKOUT_V2 = 'disabled'", "CHECKOUT_V2 = 'canary'"],
    "src/orbitops/oncall.py": ["@alice-chen", "@bob-martinez", "@platform-team"],
    "requirements.txt": ["pydantic==1.10.0", "pydantic==2.6.4"],
}


INTENSIVE_FORBIDDEN = {
    "src/orbitops/routes/report_export.py": ["/api/v1/export", "/api/v2/export", "Datadog", "Grafana Cloud"],
    "src/orbitops/client.py": ["/api/v1/export", "/api/v2/export"],
    "src/orbitops/schema.py": ["EXPORT_SCHEMA_VERSION = 'v1'", "EXPORT_SCHEMA_VERSION = 'v2'", "EXPORT_FORMAT = 'csv'"],
    "src/orbitops/serializer.py": ["schema_version': 'v1'", "schema_version': 'v2'", "'format': 'csv'", "'rows'"],
    "src/orbitops/audit.py": ["AUDIT_PROVIDER = 'datadog'", "AUDIT_PROVIDER = 'grafana-cloud'"],
    "src/orbitops/rate_limits.py": ["ADMIN_REPORTS_RATE_LIMIT = 60", "ADMIN_REPORTS_RATE_LIMIT = 90"],
    "src/orbitops/support.py": ["ops-old@example.com", "debug-secret-123"],
}


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def create_broken_repo(root: Path, *, profile: str, noise_files: int, noise_lines: int) -> None:
    write(
        root / "README.md",
        """
        # OrbitOps

        OrbitOps is an operations backend fixture for memory-layer patch testing.
        The repository intentionally contains stale decisions in active code,
        stale decisions in docs, generated distractor files, and cross-file
        contracts that must be repaired from memory.
        """,
    )
    write(root / "src/orbitops/__init__.py", "")
    write(root / "src/orbitops/routes/__init__.py", "")
    _write_core_broken_files(root)
    if profile == "intensive":
        _write_intensive_broken_files(root)
    _write_docs_and_noise(root, profile=profile, noise_files=noise_files, noise_lines=noise_lines)
    _write_tests(root, profile=profile)
    _write_contract_checker(root, profile=profile)
    write(
        root / "run_eval.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        python3 -m unittest discover -s tests
        python3 tools/check_contract.py
        """,
    )


def _write_core_broken_files(root: Path) -> None:
    write(
        root / "src/orbitops/auth.py",
        """
        class LegacyApiKeyAuth:
            def __call__(self, request):
                return bool(getattr(request, "api_key", None))
        """,
    )
    write(
        root / "src/orbitops/routes/admin_reports.py",
        """
        from orbitops.auth import LegacyApiKeyAuth

        AUTH_MIDDLEWARE = LegacyApiKeyAuth
        RATE_LIMIT = 60


        def can_access_admin_reports(request):
            return AUTH_MIDDLEWARE()(request)
        """,
    )
    write(
        root / "src/orbitops/export_jobs.py",
        """
        PAGINATION_MODE = 'offset'
        TIMEOUT_SECONDS = 5


        def export_config():
            return {
                "pagination": PAGINATION_MODE,
                "timeout": TIMEOUT_SECONDS,
            }
        """,
    )
    write(root / "src/orbitops/invoices.py", "def normalize_legacy_invoice(invoice):\n    return invoice\n")
    write(
        root / "src/orbitops/flags.py",
        """
        CHECKOUT_V2 = 'disabled'


        def enabled_flags():
            return [name for name, value in {"CHECKOUT_V2": CHECKOUT_V2}.items() if value == "enabled"]
        """,
    )
    write(
        root / "src/orbitops/oncall.py",
        """
        PAYMENT_PLATFORM_OWNER = '@alice-chen'


        def current_owner(service):
            if service == "payments-platform":
                return PAYMENT_PLATFORM_OWNER
            return None
        """,
    )
    write(root / "requirements.txt", "pydantic==1.10.0\n")


def _write_intensive_broken_files(root: Path) -> None:
    write(
        root / "src/orbitops/routes/report_export.py",
        """
        EXPORT_ENDPOINT = '/api/v1/export'
        AUDIT_PROVIDER = 'Datadog'


        def route_metadata():
            return {
                "endpoint": EXPORT_ENDPOINT,
                "audit_provider": AUDIT_PROVIDER,
            }
        """,
    )
    write(
        root / "src/orbitops/client.py",
        """
        EXPORT_ENDPOINT = '/api/v2/export'


        def export_url(base_url):
            return base_url.rstrip("/") + EXPORT_ENDPOINT
        """,
    )
    write(
        root / "src/orbitops/schema.py",
        """
        EXPORT_SCHEMA_VERSION = 'v2'
        EXPORT_FORMAT = 'csv'
        RECORD_KEY = 'rows'
        """,
    )
    write(
        root / "src/orbitops/serializer.py",
        """
        def serialize_export(records):
            return {
                'schema_version': 'v2',
                'format': 'csv',
                'rows': list(records),
            }
        """,
    )
    write(
        root / "src/orbitops/audit.py",
        """
        AUDIT_PROVIDER = 'datadog'
        AUDIT_DATASET = 'orbitops-dev'


        def audit_target():
            return f"{AUDIT_PROVIDER}:{AUDIT_DATASET}"
        """,
    )
    write(
        root / "src/orbitops/rate_limits.py",
        """
        ADMIN_REPORTS_RATE_LIMIT = 60
        EXPORT_JOBS_RATE_LIMIT = 45
        """,
    )
    write(
        root / "src/orbitops/support.py",
        """
        SUPPORT_EMAIL = 'ops-old@example.com'
        DEBUG_TOKEN = 'debug-secret-123'


        def redact_debug_token():
            return DEBUG_TOKEN
        """,
    )


def _write_docs_and_noise(root: Path, *, profile: str, noise_files: int, noise_lines: int) -> None:
    intensive_docs = ""
    if profile == "intensive":
        intensive_docs = """
        Old route note: use /api/v1/export, then /api/v2/export. Both are stale.
        Old schema note: schema_version v1/v2 and csv rows are stale.
        Old audit note: Datadog and Grafana Cloud are rejected.
        Old support note: ops-old@example.com appears in copied runbooks.
        """
    write(
        root / "docs/architecture.md",
        f"""
        # Stale Architecture Notes

        Old auth note: use LegacyApiKeyAuth for admin reports.
        Old invoice note: normalize_legacy_invoice handles invoice records.
        Old export config: offset pagination and TIMEOUT_SECONDS=5.
        {intensive_docs}
        These notes are intentionally stale. Final decisions are in memory.
        """,
    )
    for index in range(noise_files):
        lines = [
            f"# Generated distractor {index}",
            "This file contains historical notes, copied TODOs, and stale-looking strings.",
        ]
        for line in range(noise_lines):
            lines.append(
                f"NOISE_{index}_{line} = 'LegacyApiKeyAuth offset /api/v1/export "
                f"pydantic==1.10.0 @alice-chen draft-only note {line}'"
            )
        write(root / f"src/orbitops/generated/noise_{index:03d}.py", "\n".join(lines) + "\n")


def _write_tests(root: Path, *, profile: str) -> None:
    intensive_imports = ""
    intensive_tests = ""
    if profile == "intensive":
        intensive_imports = """
        from orbitops.audit import AUDIT_PROVIDER, AUDIT_DATASET, audit_target
        from orbitops.client import export_url
        from orbitops.rate_limits import ADMIN_REPORTS_RATE_LIMIT
        from orbitops.routes.report_export import EXPORT_ENDPOINT, route_metadata
        from orbitops.schema import EXPORT_SCHEMA_VERSION, EXPORT_FORMAT, RECORD_KEY
        from orbitops.serializer import serialize_export
        from orbitops.support import SUPPORT_EMAIL, redact_debug_token
        """
        intensive_tests = """
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
        """
    write(
        root / "tests/test_contract.py",
        f"""
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
        {intensive_imports}


        class Request:
            session_token = "token"


        class ContractTest(unittest.TestCase):
            def test_admin_reports_use_session_token(self):
                self.assertIs(admin_reports.AUTH_MIDDLEWARE, RequireSessionToken)
                self.assertTrue(admin_reports.can_access_admin_reports(Request()))

            def test_export_jobs_final_config(self):
                self.assertEqual(PAGINATION_MODE, "cursor")
                self.assertEqual(TIMEOUT_SECONDS, 30)
                self.assertEqual(export_config(), {{"pagination": "cursor", "timeout": 30}})

            def test_invoice_normalizer(self):
                result = normalize_active_invoice({{"id": "inv_1", "amount": "42.50"}})
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

        {intensive_tests}

        if __name__ == "__main__":
            unittest.main()
        """,
    )


def _write_contract_checker(root: Path, *, profile: str) -> None:
    forbidden = dict(CORE_FORBIDDEN)
    if profile == "intensive":
        forbidden.update(INTENSIVE_FORBIDDEN)
    content = "\n".join(
        [
            "from pathlib import Path",
            "import sys",
            "",
            "ROOT = Path(__file__).resolve().parents[1]",
            f"FORBIDDEN = {json.dumps(forbidden, indent=4)}",
            "",
            "failures = []",
            "for rel, patterns in FORBIDDEN.items():",
            "    text = (ROOT / rel).read_text(encoding='utf-8')",
            "    for pattern in patterns:",
            "        if pattern in text:",
            "            failures.append(f'{rel}: forbidden pattern remains: {pattern}')",
            "",
            "if failures:",
            "    print('\\n'.join(failures))",
            "    sys.exit(1)",
            "print('contract check passed')",
            "",
        ]
    )
    write(root / "tools/check_contract.py", content)


def expected_outcomes(profile: str) -> dict:
    expected_files = list(CORE_EXPECTED_FILES)
    required_values = list(CORE_REQUIRED_VALUES)
    forbidden_patterns = [item for values in CORE_FORBIDDEN.values() for item in values]
    if profile == "intensive":
        expected_files.extend(INTENSIVE_EXPECTED_FILES)
        required_values.extend(INTENSIVE_REQUIRED_VALUES)
        forbidden_patterns.extend(item for values in INTENSIVE_FORBIDDEN.values() for item in values)
    return {
        "test_command": "python3 -m unittest discover -s tests && python3 tools/check_contract.py",
        "profile": profile,
        "expected_files": expected_files,
        "required_values": required_values,
        "forbidden_patterns": forbidden_patterns,
    }


def task_prompt(profile: str, expected_files: list[str]) -> str:
    listed_files = "\n".join(f"- `{path}`" for path in expected_files)
    return f"""
    You are continuing work in the OrbitOps repository.

    Use the memory sessions to apply the final accepted project decisions.
    Edit the repository so all tests pass and stale/rejected implementation
    symbols are removed from active files. Do not preserve stale names for
    compatibility unless the tests explicitly require them.

    This is a {'large, ' if profile == 'intensive' else ''}multi-file patch task.
    Before finishing, inspect the tests and the contract checker, then update
    every active source/config file needed by those checks. Replace stale
    implementations with the final accepted implementations from memory. Do not
    merely delete stale code if the tests require a replacement class, function,
    constant, or config value.

    You will likely need to edit:

    {listed_files}

    The docs and generated files may intentionally contain stale historical
    notes. Active code, config, and requirements must not contain rejected
    symbols.

    Verification command:

    ```bash
    python3 -m unittest discover -s tests && python3 tools/check_contract.py
    ```
    """


def memory_sessions(profile: str) -> list[list[str]]:
    sessions = list(CORE_MEMORY_SESSIONS)
    if profile == "intensive":
        sessions.extend(INTENSIVE_MEMORY_SESSIONS)
    return sessions


def create_eval(out: Path, adapters: tuple[str, ...], *, profile: str, noise_files: int, noise_lines: int) -> None:
    if out.exists():
        raise SystemExit(f"output already exists: {out}")
    expected = expected_outcomes(profile)
    baseline = out / "baseline_repo"
    create_broken_repo(baseline, profile=profile, noise_files=noise_files, noise_lines=noise_lines)
    (baseline / "run_eval.sh").chmod(0o755)

    workspaces = out / "workspaces"
    for adapter in adapters:
        target = workspaces / adapter
        shutil.copytree(baseline, target)
        (target / "run_eval.sh").chmod(0o755)

    write(out / "task_prompt.md", task_prompt(profile, expected["expected_files"]))
    write(
        out / "memory_sessions.md",
        "\n\n".join(
            f"## Session {index}\n\n" + "\n".join(f"- {event}" for event in session)
            for index, session in enumerate(memory_sessions(profile), start=1)
        ),
    )
    write(
        out / "README.md",
        f"""
        # Repo Patch Memory Eval

        Profile: `{profile}`

        This folder contains identical broken repo workspaces for each memory
        layer. Give every agent the same `task_prompt.md`, the same
        `memory_sessions.md`, and only its own workspace path.

        Workspaces:

        {chr(10).join(f"- `workspaces/{adapter}`" for adapter in adapters)}

        Score each run with test success, stale cleanup, expected file touches,
        total input/output tokens, number of turns/tool calls, wall-clock time,
        and diff size.
        """,
    )
    write(out / "expected_outcomes.json", json.dumps(expected, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=f"repoPatchMemoryEval/run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Output directory to create.",
    )
    parser.add_argument("--adapters", default=",".join(DEFAULT_ADAPTERS))
    parser.add_argument("--profile", choices=("basic", "intensive"), default="intensive")
    parser.add_argument("--noise-files", type=int, default=40)
    parser.add_argument("--noise-lines", type=int, default=25)
    args = parser.parse_args()

    adapters = tuple(item.strip() for item in args.adapters.split(",") if item.strip())
    if not adapters:
        raise SystemExit("--adapters must contain at least one name")
    out = Path(args.out)
    create_eval(out, adapters, profile=args.profile, noise_files=args.noise_files, noise_lines=args.noise_lines)
    print(f"created repo patch memory eval at {out}")
    print(f"profile: {args.profile}")
    print(f"task prompt: {out / 'task_prompt.md'}")
    print(f"memory sessions: {out / 'memory_sessions.md'}")
    print(f"workspaces: {out / 'workspaces'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
