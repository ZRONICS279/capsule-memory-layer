class RequireSessionToken:
    def __call__(self, request):
        return bool(getattr(request, 'session_token', None) or getattr(request, 'headers', {}).get('session_token'))
