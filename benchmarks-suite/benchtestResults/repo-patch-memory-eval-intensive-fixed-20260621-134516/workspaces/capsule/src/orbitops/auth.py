class RequireSessionToken:
    def __call__(self, request):
        return bool(getattr(request, "session_token", None))


EXPORT_SCHEMA_VERSION = "v3"
EXPORT_FORMAT = "parquet"
RECORD_KEY = "records"
