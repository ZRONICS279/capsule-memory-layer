"""Deterministic data pools for generating distinct benchmark variants.

Each pool contains 10+ entries so ``--cases-per-family 7`` (or higher)
draws genuinely different entity names, values, and patterns per variant
rather than repeating the same scenario with extra noise.
"""

from __future__ import annotations

import random


# ---------------------------------------------------------------------------
# Cross-session coding pools
# ---------------------------------------------------------------------------

AUTH_CLASSES: list[tuple[str, str]] = [
    ("LegacyApiKeyAuth", "RequireSessionToken"),
    ("BasicAuthMiddleware", "OAuthBearerToken"),
    ("HmacSignatureAuth", "JwtValidationMiddleware"),
    ("CookieSessionAuth", "MtlsCertificateAuth"),
    ("SharedSecretAuth", "SamlAssertionMiddleware"),
    ("ApiGatewayAuth", "OpenIdConnectValidator"),
    ("LdapBindAuth", "PasskeyWebAuthnAuth"),
    ("DigestAccessAuth", "ScopedServiceToken"),
    ("StaticTokenAuth", "RotatingKeyMiddleware"),
    ("IpWhitelistAuth", "ZeroTrustPolicyAuth"),
]

CONFIG_KEYS: list[tuple[str, str, str]] = [
    ("BILLING_RETRY_COUNT", "MAX_RETRIES_3", "PAYMENT_RETRY_LIMIT_0"),
    ("DB_POOL_SIZE", "CONN_POOL_MAX_5", "POOL_CONNECTIONS_20"),
    ("CACHE_TTL_SECONDS", "CACHE_EXPIRY_60", "CACHE_DURATION_300"),
    ("RATE_LIMIT_RPM", "THROTTLE_MAX_100", "REQUEST_CAP_500"),
    ("LOG_RETENTION_DAYS", "KEEP_LOGS_7", "AUDIT_LOG_RETENTION_90"),
    ("BATCH_SIZE", "CHUNK_SIZE_50", "PROCESSING_BATCH_200"),
    ("SESSION_TIMEOUT_MIN", "IDLE_TIMEOUT_15", "SESSION_EXPIRY_60"),
    ("WEBHOOK_RETRY_MAX", "RETRY_COUNT_3", "DELIVERY_ATTEMPTS_5"),
    ("UPLOAD_MAX_MB", "FILE_LIMIT_10", "UPLOAD_CEILING_100"),
    ("WORKER_CONCURRENCY", "THREAD_COUNT_4", "WORKER_POOL_16"),
]

ROUTES: list[tuple[str, str, str]] = [
    ("/api/v1/export", "/api/v2/reports/export", "/api/v3/reports/{id}/export"),
    ("/api/v1/users/list", "/api/v2/users", "/api/v3/accounts/{id}/members"),
    ("/api/billing/charge", "/api/v2/payments/charge", "/api/v3/invoices/{id}/pay"),
    ("/api/v1/search", "/api/v2/query", "/api/v3/catalog/search"),
    ("/api/events/log", "/api/v2/events/ingest", "/api/v3/telemetry/events"),
    ("/api/v1/notify", "/api/v2/notifications/send", "/api/v3/channels/{id}/notify"),
    ("/api/auth/login", "/api/v2/auth/token", "/api/v3/sessions/create"),
    ("/api/v1/upload", "/api/v2/files/upload", "/api/v3/storage/{bucket}/put"),
    ("/api/config/get", "/api/v2/settings", "/api/v3/tenants/{id}/config"),
    ("/api/v1/health", "/api/v2/status", "/api/v3/diagnostics/health"),
]

OWNERS: list[tuple[str, ...]] = [
    ("alice-chen", "bob-martinez", "platform-team", "carol-wu"),
    ("dana-kim", "eric-johansson", "infra-squad", "fatima-ali"),
    ("george-patel", "hana-ito", "oncall-rotation", "ivan-novak"),
    ("julia-okafor", "kevin-lee", "sre-team", "lena-schmidt"),
    ("marcus-davis", "nina-berg", "devops-guild", "omar-hassan"),
    ("priya-sharma", "quinn-taylor", "security-team", "rosa-lopez"),
    ("sam-wright", "tina-chen", "reliability-pod", "umar-khan"),
    ("vera-petrov", "walt-jones", "backend-crew", "xena-park"),
    ("yuri-tanaka", "zara-ahmed", "core-infra", "alex-rivera"),
    ("blake-moore", "casey-lin", "site-reliability", "drew-foster"),
]

VERSIONS: list[tuple[str, str, str, str]] = [
    ("pydantic", "1.10.0", "2.6.4", "2.7.0"),
    ("sqlalchemy", "1.4.49", "2.0.25", "2.0.30"),
    ("fastapi", "0.95.0", "0.108.0", "0.111.0"),
    ("django", "3.2.25", "4.2.11", "5.0.6"),
    ("flask", "2.3.3", "3.0.0", "3.0.3"),
    ("httpx", "0.24.1", "0.26.0", "0.27.0"),
    ("celery", "5.2.7", "5.3.4", "5.4.0"),
    ("redis", "4.5.5", "5.0.1", "5.0.7"),
    ("boto3", "1.28.0", "1.34.0", "1.34.100"),
    ("numpy", "1.24.4", "1.26.4", "2.0.0"),
]

FLAG_NAMES: list[str] = [
    "new_checkout_flow",
    "dark_mode_v2",
    "async_billing_pipeline",
    "graphql_gateway",
    "realtime_notifications",
    "multi_tenant_isolation",
    "ai_search_ranking",
    "progressive_image_loading",
    "zero_downtime_migration",
    "feature_usage_analytics",
]

FUNCTION_NAMES: list[tuple[str, str]] = [
    ("normalize_legacy_invoice", "normalize_active_invoice_0"),
    ("parse_old_format", "parse_current_format_0"),
    ("validate_v1_schema", "validate_v2_schema_0"),
    ("compute_tax_legacy", "compute_tax_current_0"),
    ("render_classic_template", "render_modern_template_0"),
    ("serialize_compat", "serialize_standard_0"),
    ("decode_legacy_token", "decode_current_token_0"),
    ("route_legacy_handler", "route_current_handler_0"),
    ("format_old_response", "format_current_response_0"),
    ("connect_legacy_db", "connect_current_db_0"),
]

FILE_PATHS: list[tuple[str, str]] = [
    ("src/auth/middleware.py", "src/api/routes/admin_reports.py"),
    ("src/auth/oauth.py", "src/api/routes/user_sessions.py"),
    ("src/auth/jwt_validator.py", "src/api/routes/service_tokens.py"),
    ("src/auth/mtls.py", "src/api/routes/certificate_auth.py"),
    ("src/auth/saml.py", "src/api/routes/sso_login.py"),
    ("src/auth/passkey.py", "src/api/routes/webauthn.py"),
    ("src/auth/zero_trust.py", "src/api/routes/policy_check.py"),
    ("src/auth/scoped_token.py", "src/api/routes/token_exchange.py"),
    ("src/auth/rotating_key.py", "src/api/routes/key_rotation.py"),
    ("src/auth/ip_policy.py", "src/api/routes/access_control.py"),
]


# ---------------------------------------------------------------------------
# Chat memory pools
# ---------------------------------------------------------------------------

CODING_LANGUAGES: list[tuple[str, str]] = [
    ("Python", "TypeScript"),
    ("JavaScript", "Rust"),
    ("Java", "Go"),
    ("Ruby", "Elixir"),
    ("PHP", "Kotlin"),
    ("C#", "Swift"),
    ("Perl", "Haskell"),
    ("Scala", "Zig"),
    ("Lua", "Dart"),
    ("R", "Julia"),
]

TRAVEL_DESTINATIONS: list[tuple[str, str]] = [
    ("Kyoto", "Hotel Granvia Kyoto"),
    ("Barcelona", "Hotel Arts Barcelona"),
    ("Lisbon", "Pestana Palace"),
    ("Cape Town", "The Silo Hotel"),
    ("Reykjavik", "Ion Adventure Hotel"),
    ("Marrakech", "La Mamounia"),
    ("Seoul", "Signiel Seoul"),
    ("Buenos Aires", "Alvear Palace Hotel"),
    ("Prague", "Augustine Hotel"),
    ("Hanoi", "Sofitel Legend Metropole"),
]

DIET_PREFERENCES: list[tuple[str, str]] = [
    ("vegan", "keto"),
    ("vegetarian", "paleo"),
    ("pescatarian", "carnivore"),
    ("gluten-free", "Mediterranean"),
    ("raw food", "whole30"),
    ("macrobiotic", "DASH"),
    ("low-carb", "high-protein"),
    ("fruitarian", "flexitarian"),
    ("anti-inflammatory", "FODMAP"),
    ("Ayurvedic", "Nordic"),
]

APP_NAMES: list[tuple[str, str]] = [
    ("AtlasDesk", "Northstar CRM"),
    ("TaskFlow", "Orbit PM"),
    ("DataPulse", "InsightEngine"),
    ("CodeVault", "GitForge"),
    ("ChatNest", "ThreadSpace"),
    ("PixelDraft", "CanvasLab"),
    ("BudgetOwl", "FinTrack Pro"),
    ("MealMate", "NourishAI"),
    ("FitLog", "BodyMetrics"),
    ("TravelWise", "JourneyPlan"),
]

NOVEL_ELEMENTS: list[tuple[str, str, str]] = [
    ("Mira", "memory loss", "childhood memory"),
    ("Kai", "dream walking", "recurring nightmare"),
    ("Zara", "time loops", "forgotten conversation"),
    ("Theo", "emotion theft", "first love"),
    ("Luna", "shadow binding", "mother's lullaby"),
    ("Ren", "name magic", "true name"),
    ("Sage", "ink sorcery", "unfinished letter"),
    ("Nyx", "mirror worlds", "reflection self"),
    ("Ash", "bone singing", "ancestor's voice"),
    ("Ivy", "root networks", "family tree"),
]

PHONE_NUMBERS: list[str] = [
    "555-1234", "555-6789", "555-4321", "555-8765", "555-2468",
    "555-1357", "555-9876", "555-3141", "555-2718", "555-1618",
]

MEDICAL_TOPICS: list[tuple[str, str]] = [
    ("medical anxiety", "workplace email"),
    ("chronic fatigue", "quarterly report"),
    ("sleep disorder", "team standup notes"),
    ("allergy management", "sprint planning"),
    ("physical therapy", "client proposal"),
    ("medication schedule", "budget review"),
    ("dental phobia", "product roadmap"),
    ("blood pressure", "performance review"),
    ("migraine triggers", "meeting agenda"),
    ("back pain exercises", "project timeline"),
]


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class VariantSelector:
    """Deterministic pool selection keyed by (family, variant)."""

    def __init__(self, family: str, variant: int, *, seed_base: int = 42_000) -> None:
        self.rng = random.Random(seed_base + hash(family) + variant * 7919)
        self.variant = variant

    def pick(self, pool: list, idx: int = 0) -> object:
        """Return pool[variant % len(pool)], no randomness needed for simple round-robin."""
        return pool[(self.variant + idx) % len(pool)]

    def pick_random(self, pool: list) -> object:
        return self.rng.choice(pool)

    def noise_count(self, base: int = 3) -> int:
        return base + self.variant * 2

    def session_noise(self, label: str, count: int | None = None) -> tuple[str, ...]:
        n = count if count is not None else self.noise_count()
        topics = ["architecture review", "CI pipeline config", "deployment checklist",
                   "code review feedback", "sprint retro notes", "onboarding docs",
                   "monitoring alert tuning", "dependency audit", "API versioning discussion",
                   "database index analysis", "load test results", "feature planning"]
        return tuple(
            f"SESSION NOISE {label} note={i} variant={self.variant}: "
            f"{self.rng.choice(topics)} — unrelated context."
            for i in range(n)
        )

    def chat_noise(self, label: str, count: int | None = None) -> tuple[str, ...]:
        n = count if count is not None else self.noise_count()
        topics = ["grocery list", "podcast recommendation", "weekend plans",
                   "music playlist", "calendar reminder", "birthday gift idea",
                   "movie review", "recipe bookmark", "workout schedule",
                   "book recommendation", "travel wishlist", "home renovation"]
        return tuple(
            f"CHAT NOISE {label} note={i} variant={self.variant}: "
            f"unrelated reminders about {self.rng.choice(topics)}."
            for i in range(n)
        )
