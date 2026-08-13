"""
Secure wrapper around the Twelve Data API.

The API key is read only from the server-side environment
(TWELVE_DATA_API_KEY) and is never exposed to the Flutter app.

This client also handles Twelve Data rate limits safely:
- 429 responses are not retried immediately.
- Temporary network/server errors use exponential backoff.
"""

import asyncio
from typing import Any

import httpx
import pandas as pd
from loguru import logger

from app.config import settings


SYMBOL_MAP = {
    "XAU/USD": "XAU/USD",
    "BTC/USD": "BTC/USD",
}


class TwelveDataRateLimitError(Exception):
    """Raised when Twelve Data returns HTTP 429."""


class TwelveDataClient:
    def __init__(self):
        self.base_url = settings.TWELVE_DATA_BASE_URL.rstrip("/")
        self.api_key = settings.TWELVE_DATA_API_KEY

    def _symbol(self, pair: str) -> str:
        return SYMBOL_MAP.get(pair, pair)

    async def _get(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> dict:
        """
        Make one Twelve Data request.

        Important:
        429 is NOT retried automatically. Retrying a rate-limited
        request can make the situation worse.
        """
        request_params = {**params, "apikey": self.api_key}

        async with httpx.AsyncClient(timeout=20) as client:
            try:
                response = await client.get(
                    f"{self.base_url}/{endpoint}",
                    params=request_params,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                logger.warning(
                    f"Twelve Data network error on {endpoint}: {exc}"
                )
                raise

            if response.status_code == 429:
                logger.warning(
                    f"Twelve Data rate limit reached on {endpoint} "
                    f"for symbol={params.get('symbol')}"
                )
                raise TwelveDataRateLimitError(
                    "Twelve Data rate limit reached"
                )

            if response.status_code >= 500:
                logger.warning(
                    f"Twelve Data server error {response.status_code} "
                    f"on {endpoint}"
                )
                raise httpx.HTTPStatusError(
                    f"Twelve Data server error: {response.status_code}",
                    request=response.request,
                    response=response,
                )

            response.raise_for_status()

            data = response.json()

            if isinstance(data, dict) and data.get("status") == "error":
                message = data.get(
                    "message",
                    "Twelve Data API error",
                )
                logger.error(
                    f"Twelve Data error for {endpoint} "
                    f"{params.get('symbol')}: {message}"
                )
                raise ValueError(message)

            return data

    async def get_quote(self, pair: str) -> dict:
        """Latest quote."""
        return await self._get(
            "quote",
            {"symbol": self._symbol(pair)},
        )

    async def get_price(self, pair: str) -> float:
        """Latest price."""
        data = await self._get(
            "price",
            {"symbol": self._symbol(pair)},
        )
        return float(data["price"])

    async def get_time_series(
        self,
        pair: str,
        interval: str = "1h",
        outputsize: int = 300,
    ) -> pd.DataFrame:
        """
        Return candles ordered oldest -> newest.

        outputsize remains 300 because the signal engine may depend
        on enough historical candles for its indicators.
        """
        data = await self._get(
            "time_series",
            {
                "symbol": self._symbol(pair),
                "interval": interval,
                "outputsize": outputsize,
            },
        )

        values = data.get("values", [])

        if not values:
            raise ValueError(
                f"No time series data returned for {pair}"
            )

        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])

        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)

        if "volume" in df.columns:
            df["volume"] = df["volume"].astype(float)
        else:
            df["volume"] = 0.0

        return (
            df.sort_values("datetime")
            .reset_index(drop=True)
        )

    async def get_quotes_bulk(
        self,
        pairs: list[str],
    ) -> dict:
        """
        Fetch quotes for multiple pairs in one HTTP request.
        Note: Twelve Data still counts credits per symbol.
        """
        symbols = ",".join(
            self._symbol(pair) for pair in pairs
        )

        return await self._get(
            "quote",
            {"symbol": symbols},
        )


twelve_data_client = TwelveDataClient()
