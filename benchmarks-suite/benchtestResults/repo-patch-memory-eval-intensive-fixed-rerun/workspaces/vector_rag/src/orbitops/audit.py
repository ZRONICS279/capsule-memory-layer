AUDIT_PROVIDER = 'internal'
AUDIT_DATASET = 'orbitops'


def audit_target():
    return f'{AUDIT_PROVIDER}:{AUDIT_DATASET}'
