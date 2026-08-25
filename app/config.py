"""
Central application configuration.
All secrets are read from environment variables ONLY.
Never hardcode API keys, DB passwords, or JWT secrets here.
"""

from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    # =========================================================
    # FCS API - PRIMARY MARKET DATA PROVIDER
    # =========================================================

    FCS_API_KEY: str = ""
    FCS_BASE_URL: str = "https://api-v4.fcsapi.com"

    # =========================================================
    # TWELVE DATA - FALLBACK MARKET DATA PROVIDER
    # =========================================================

    TWELVE_DATA_API_KEY: str
    TWELVE_DATA_BASE_URL: str = "https://api.twelvedata.com"

    # =========================================================
    # DATABASE
    # =========================================================

    DATABASE_URL: str

    # =========================================================
    # ADMIN BOOTSTRAP
    # =========================================================

    ADMIN_EMAIL: str = "admin@example.com"
    ADMIN_PASSWORD: str = "change_me"
    ADMIN_API_KEY: str = "change_me"

    # =========================================================
    # FIREBASE
    # =========================================================

    FIREBASE_CREDENTIALS_PATH: str = ""

    FIREBASE_WEB_API_KEY: str = ""

    # =========================================================
    # APP
    # =========================================================

    ENVIRONMENT: str = "development"

    ALLOWED_ORIGINS: str = "http://localhost:3000"

    SIGNAL_POLL_INTERVAL_SECONDS: int = 300

    MIN_RISK_REWARD_RATIO: float = 2.0

    MIN_CONFIDENCE_TO_PUBLISH: int = 65

    RATE_LIMIT_PER_MINUTE: int = 60

    # =========================================================
    # CORS
    # =========================================================

    @property
    def cors_origins(self) -> List[str]:
        return [
            o.strip()
            for o in self.ALLOWED_ORIGINS.split(",")
            if o.strip()
        ]


settings = Settings()


# =============================================================
# SUPPORTED TRADING PAIRS
# =============================================================

SUPPORTED_PAIRS = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "USD/CAD",
    "AUD/USD",
    "NZD/USD",
    "EUR/JPY",
    "GBP/JPY",
    "XAU/USD",
    "BTC/USD",
]
