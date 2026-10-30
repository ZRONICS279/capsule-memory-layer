AUDIT_PROVIDER = 'honeycomb'
AUDIT_DATASET = 'orbitops-prod'


def audit_target():
    return f"{AUDIT_PROVIDER}:{AUDIT_DATASET}"
