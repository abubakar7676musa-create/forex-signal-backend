"""
Secure Twelve Data API client.

Goals:
- Never expose the API key to the Flutter app.
- Handle HTTP 429 without retry storms.
- Respect Twelve Data's minute-based API credit limits.
- Read API credit information from response headers.
- Keep the existing public methods compatible with the app.
"""

import asyncio
import time
from collections import deque
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
    """Raised when Twelve Data rejects a request because of rate limits."""


class TwelveDataClient:
    def __init__(self):
        self.base_url = settings.TWELVE_DATA_BASE_URL.rstrip("/")
        self.api_key = settings.TWELVE_DATA_API_KEY

        # Conservative default for the Free/Basic plan.
        # Twelve Data Basic currently allows 8 API credits/minute.
        #
        # We intentionally reserve 1 credit so the application does
        # not constantly operate exactly at the provider's hard limit.
        self.max_credits_per_minute = 7

        # Local rolling-window limiter.
        # Stores (timestamp, estimated credits).
        self._credit_history: deque[tuple[float, int]] = deque()

        # Prevent multiple scheduler tasks from passing the limiter
        # simultaneously.
        self._rate_lock = asyncio.Lock()

    def _symbol(self, pair: str) -> str:
        return SYMBOL_MAP.get(pair, pair)

    def _cleanup_credit_history(self) -> None:
        now = time.monotonic()

        while self._credit_history:
            timestamp, _credits = self._credit_history[0]

            if now - timestamp >= 60:
                self._credit_history.popleft()
            else:
                break

    async def _wait_for_credit_slot(
        self,
        estimated_credits: int = 1,
    ) -> None:
        """
        Wait until the local rolling-window limiter allows the request.

        We use a conservative 7-credit/minute ceiling for the Free/Basic
        plan instead of attempting to consume the full 8-credit quota.
        """

        async with self._rate_lock:
            while True:
                self._cleanup_credit_history()

                used = sum(
                    credits
                    for _timestamp, credits in self._credit_history
                )

                if (
                    used + estimated_credits
                    <= self.max_credits_per_minute
                ):
                    self._credit_history.append(
                        (time.monotonic(), estimated_credits)
                    )
                    return

                oldest_timestamp, _ = self._credit_history[0]

                wait_seconds = max(
                    0.5,
                    60
                    - (time.monotonic() - oldest_timestamp)
                    + 0.25,
                )

                logger.info(
                    "Twelve Data local rate limiter: "
                    f"{used}/{self.max_credits_per_minute} credits used. "
                    f"Waiting {wait_seconds:.1f}s."
                )

                await asyncio.sleep(wait_seconds)

    def _sync_credit_headers(
        self,
        response: httpx.Response,
    ) -> None:
        """
        Read Twelve Data credit headers.

        We do not rely on these headers as the only protection because
        the local limiter already prevents request bursts.
        """

        credits_left = response.headers.get(
            "api-credits-left"
        )

        credits_used = response.headers.get(
            "api-credits-used"
        )

        if credits_left is not None:
            logger.debug(
                "Twelve Data credits: "
                f"used={credits_used}, "
                f"left={credits_left}"
            )

    async def _get(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> dict:
        """
        Perform one request safely.

        Important:
        - 429 is NOT retried immediately.
        - Network failures are raised normally.
        - HTTP 5xx errors are raised normally.
        - API-level errors are converted to ValueError.
        """

        # For the endpoints currently used by this application,
        # one symbol/request is conservatively counted as one credit.
        await self._wait_for_credit_slot(1)

        request_params = {
            **params,
            "apikey": self.api_key,
        }

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10,
                read=20,
                write=10,
                pool=10,
            )
        ) as client:

            try:
                response = await client.get(
                    f"{self.base_url}/{endpoint}",
                    params=request_params,
                )

            except httpx.TimeoutException as exc:
                logger.warning(
                    f"Twelve Data timeout on {endpoint}: {exc}"
                )
                raise

            except httpx.NetworkError as exc:
                logger.warning(
                    f"Twelve Data network error on "
                    f"{endpoint}: {exc}"
                )
                raise

            self._sync_credit_headers(response)

            if response.status_code == 429:
                logger.warning(
                    "Twelve Data returned HTTP 429. "
                    "The provider quota has been reached. "
                    "This request will NOT be retried immediately."
                )

                # Remove the local reservation because the provider
                # rejected the request. This prevents our local limiter
                # from thinking the rejected request consumed a credit.
                async with self._rate_lock:
                    self._cleanup_credit_history()

                    if self._credit_history:
                        last_timestamp, last_credits = (
                            self._credit_history[-1]
                        )

                        if (
                            time.monotonic()
                            - last_timestamp
                            < 2
                            and last_credits == 1
                        ):
                            self._credit_history.pop()

                raise TwelveDataRateLimitError(
                    "Twelve Data API rate limit reached."
                )

            if 500 <= response.status_code <= 599:
                logger.warning(
                    "Twelve Data server error "
                    f"{response.status_code} on {endpoint}."
                )

                response.raise_for_status()

            response.raise_for_status()

            try:
                data = response.json()
            except ValueError as exc:
                logger.error(
                    f"Invalid JSON response from Twelve Data "
                    f"endpoint={endpoint}"
                )
                raise ValueError(
                    "Invalid JSON response from Twelve Data"
                ) from exc

            if isinstance(data, dict):
                if data.get("status") == "error":
                    message = data.get(
                        "message",
                        "Twelve Data API error",
                    )

                    logger.error(
                        f"Twelve Data API error on "
                        f"{endpoint}: {message}"
                    )

                    raise ValueError(message)

            return data

    async def get_quote(self, pair: str) -> dict:
        """Return the latest quote for a pair."""
        return await self._get(
            "quote",
            {
                "symbol": self._symbol(pair),
            },
        )

    async def get_price(self, pair: str) -> float:
        """Return the latest price for a pair."""
        data = await self._get(
            "price",
            {
                "symbol": self._symbol(pair),
            },
        )

        return float(data["price"])

    async def get_time_series(
        self,
        pair: str,
        interval: str = "1h",
        outputsize: int = 300,
    ) -> pd.DataFrame:
        """
        Return OHLCV data ordered oldest -> newest.

        The method signature is intentionally unchanged so the existing
        signal engine does not need to be modified.
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

        if "datetime" not in df.columns:
            raise ValueError(
                f"Twelve Data response for {pair} "
                "does not contain datetime values."
            )

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
        )

        df = df.dropna(subset=["datetime"])

        for column in [
            "open",
            "high",
            "low",
            "close",
        ]:
            if column not in df.columns:
                raise ValueError(
                    f"Missing required column '{column}' "
                    f"for {pair}"
                )

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df = df.dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )

        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(
                df["volume"],
                errors="coerce",
            ).fillna(0.0)
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
        Fetch multiple quotes using one HTTP request.

        Note:
        Twelve Data still counts credits per symbol in a batch.
        Batch requests reduce HTTP overhead but do not magically
        reduce the credit cost.
        """

        symbols = ",".join(
            self._symbol(pair)
            for pair in pairs
        )

        return await self._get(
            "quote",
            {
                "symbol": symbols,
            },
        )


twelve_data_client = TwelveDataClient()
