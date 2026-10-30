class LegacyApiKeyAuth:
    def __call__(self, request):
        return bool(getattr(request, "api_key", None))
