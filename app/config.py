"""
Central application configuration.
All secrets are read from environment variables ONLY.
Never hardcode API keys, DB passwords, or JWT secrets here.
"""
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # Twelve Data
    TWELVE_DATA_API_KEY: str
    TWELVE_DATA_BASE_URL: str = "https://api.twelvedata.com"

    # Database
    DATABASE_URL: str

    # Admin bootstrap (used once on startup to ensure an admin user exists)
    ADMIN_EMAIL: str = "admin@example.com"
    ADMIN_PASSWORD: str = "change_me"
    ADMIN_API_KEY: str = "change_me"

    # Firebase
    # Service account JSON (server-side, secret) — used by the Admin SDK to create
    # users, verify ID tokens, and send FCM push notifications.
    FIREBASE_CREDENTIALS_PATH: str = ""
    # Firebase project's Web API key (the same key used in any web/mobile Firebase
    # config — not a secret in the same sense as the service account, but still
    # kept out of source control). Required for the /auth/login and /auth/refresh
    # endpoints, which validate passwords via the Identity Toolkit REST API.
    FIREBASE_WEB_API_KEY: str = ""

    # App
    ENVIRONMENT: str = "development"
    ALLOWED_ORIGINS: str = "http://localhost:3000"
    SIGNAL_POLL_INTERVAL_SECONDS: int = 300
    MIN_RISK_REWARD_RATIO: float = 2.0
    MIN_CONFIDENCE_TO_PUBLISH: int = 65
    RATE_LIMIT_PER_MINUTE: int = 60

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]


settings = Settings()

SUPPORTED_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "AUD/USD",
    "NZD/USD", "EUR/JPY", "GBP/JPY", "XAU/USD", "BTC/USD",
]
