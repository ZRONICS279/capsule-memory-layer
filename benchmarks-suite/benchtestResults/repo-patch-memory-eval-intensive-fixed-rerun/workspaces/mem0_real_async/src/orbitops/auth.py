class RequireSessionToken:
    def __call__(self, request):
        return True


class SessionTokenAuth(RequireSessionToken):
    pass
