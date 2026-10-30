from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "src/orbitops/auth.py": ["LegacyApiKeyAuth"],
    "src/orbitops/routes/admin_reports.py": ["LegacyApiKeyAuth"],
    "src/orbitops/export_jobs.py": ["PAGINATION_MODE = 'offset'", "TIMEOUT_SECONDS = 5"],
    "src/orbitops/invoices.py": ["normalize_legacy_invoice"],
    "src/orbitops/flags.py": ["CHECKOUT_V2 = 'disabled'", "CHECKOUT_V2 = 'canary'"],
    "src/orbitops/oncall.py": ["@alice-chen", "@bob-martinez", "@platform-team"],
    "requirements.txt": ["pydantic==1.10.0", "pydantic==2.6.4"],
}

failures = []
for rel, patterns in FORBIDDEN.items():
    text = (ROOT / rel).read_text(encoding="utf-8")
    for pattern in patterns:
        if pattern in text:
            failures.append(f"{rel}: forbidden pattern remains: {pattern}")

if failures:
    print("\n".join(failures))
    sys.exit(1)
print("contract check passed")
