You are continuing work in the OrbitOps repository.

    Use the memory sessions to apply the final accepted project decisions.
    Edit the repository so all tests pass and stale/rejected implementation
    symbols are removed from active files. Do not preserve stale names for
    compatibility unless the tests explicitly require them.

    This is a large, multi-file patch task.
    Before finishing, inspect the tests and the contract checker, then update
    every active source/config file needed by those checks. Replace stale
    implementations with the final accepted implementations from memory. Do not
    merely delete stale code if the tests require a replacement class, function,
    constant, or config value.

    You will likely need to edit:

    - `requirements.txt`
- `src/orbitops/auth.py`
- `src/orbitops/routes/admin_reports.py`
- `src/orbitops/export_jobs.py`
- `src/orbitops/invoices.py`
- `src/orbitops/flags.py`
- `src/orbitops/oncall.py`
- `src/orbitops/routes/report_export.py`
- `src/orbitops/client.py`
- `src/orbitops/schema.py`
- `src/orbitops/serializer.py`
- `src/orbitops/audit.py`
- `src/orbitops/rate_limits.py`
- `src/orbitops/support.py`

    The docs and generated files may intentionally contain stale historical
    notes. Active code, config, and requirements must not contain rejected
    symbols.

    Verification command:

    ```bash
    python3 -m unittest discover -s tests && python3 tools/check_contract.py
    ```
