# Security

This project stores agent memory. Treat memory as sensitive user/application data.

Current status:

- local JSON and SQLite stores are not encrypted
- no remote server is included
- no telemetry is included
- soft-token training code uses local files/models only

Before production use, add:

- encryption at rest where needed
- tenant isolation
- PII redaction hooks
- retention/deletion policies
- audit logging
- access control around capsule inspection

Report security issues privately once a project maintainer contact exists.

