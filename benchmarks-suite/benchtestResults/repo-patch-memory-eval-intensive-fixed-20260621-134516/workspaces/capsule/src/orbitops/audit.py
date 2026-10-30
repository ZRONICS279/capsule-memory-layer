AUDIT_PROVIDER = "honeycomb"
AUDIT_DATASET = "orbitops-dev"


def audit_target():
    return f"{AUDIT_PROVIDER}:{AUDIT_DATASET}"
