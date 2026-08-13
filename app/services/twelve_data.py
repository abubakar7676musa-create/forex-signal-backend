
"""
Secure Twelve Data API client.

Goals:
- Never expose the API key to the Flutter app.
- Prevent HTTP 429 retry storms.
- Respect Twelve Data minute-based API credit limits.
- Dynamically learn the real quota from response headers.
- Keep existing public methods compatible.
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
    """Raised when Twelve Data quota is exhausted."""


class TwelveDataClient:
    def __init__(self):
        self.base_url = settings.TWELVE_DATA_BASE_URL.rstrip("/")
        self.api_key = settings.TWELVE_DATA_API_KEY

        # Your current Free/Basic plan is 8 credits/minute.
        #
        # Keep one credit in reserve so the application does not
        # continuously operate at the provider's absolute limit.
        self.default_credits_per_minute = 8
        self.safety_reserve = 1

        self.max_credits_per_minute = (
            self.default_credits_per_minute
            - self.safety_reserve
        )

        # Local rolling credit history.
        #
        # (monotonic_timestamp, estimated_credits)
        self._credit_history: deque[tuple[float, int]] = deque()

        # Provider-reported state.
        self._provider_credits_left: int | None = None
        self._provider_credits_used: int | None = None
        self._provider_limit: int | None = None

        # Prevent concurrent requests from bypassing the limiter.
        self._rate_lock = asyncio.Lock()

        # Reuse one HTTP connection pool instead of creating a client
        # for every single request.
        self._client: httpx.AsyncClient | None = None

        # When the provider returns 429, do not hammer it again.
        self._blocked_until: float = 0.0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10,
                    read=20,
                    write=10,
                    pool=10,
                ),
                limits=httpx.Limits(
                    max_connections=5,
                    max_keepalive_connections=5,
                ),
            )

        return self._client

    async def close(self) -> None:
        """Close the shared HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

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

    def _seconds_until_next_minute(self) -> float:
        """
        Twelve Data restores the full API quota at the start of
        a new minute.

        Use wall-clock time here because the provider's reset is
        minute based.
        """
        now = time.time()
        seconds = 60 - (now % 60)

        # Small safety buffer for clock/network differences.
        return max(1.0, seconds + 0.5)

    async def _wait_for_credit_slot(
        self,
        estimated_credits: int = 1,
    ) -> None:
        """
        Wait until it is safe to send another API request.
        """

        while True:
            async with self._rate_lock:
                now = time.monotonic()

                # If a previous 429 blocked the client, wait until
                # the provider quota should have reset.
                if now < self._blocked_until:
                    wait_seconds = (
                        self._blocked_until - now
                    )

                    logger.warning(
                        "Twelve Data client temporarily blocked. "
                        f"Waiting {wait_seconds:.1f}s."
                    )
                else:
                    self._cleanup_credit_history()

                    used_local = sum(
                        credits
                        for _timestamp, credits
                        in self._credit_history
                    )

                    provider_left = self._provider_credits_left

                    # If Twelve Data explicitly told us that no credits
                    # remain, wait for the next minute.
                    if (
                        provider_left is not None
                        and provider_left < estimated_credits
                    ):
                        wait_seconds = (
                            self._seconds_until_next_minute()
                        )

                        logger.info(
                            "Twelve Data provider credits exhausted. "
                            f"left={provider_left}. "
                            f"Waiting {wait_seconds:.1f}s "
                            "for quota reset."
                        )

                    elif (
                        used_local + estimated_credits
                        <= self.max_credits_per_minute
                    ):
                        self._credit_history.append(
                            (
                                time.monotonic(),
                                estimated_credits,
                            )
                        )
                        return

                    else:
                        # Local rolling window is full.
                        if self._credit_history:
                            oldest_timestamp, _ = (
                                self._credit_history[0]
                            )

                            wait_seconds = max(
                                1.0,
                                60
                                - (
                                    time.monotonic()
                                    - oldest_timestamp
                                )
                                + 0.5,
                            )
                        else:
                            wait_seconds = (
                                self._seconds_until_next_minute()
                            )

                        logger.info(
                            "Twelve Data local limiter: "
                            f"{used_local}/"
                            f"{self.max_credits_per_minute} "
                            "credits reserved. "
                            f"Waiting {wait_seconds:.1f}s."
                        )

            await asyncio.sleep(wait_seconds)

    def _sync_credit_headers(
        self,
        response: httpx.Response,
    ) -> None:
        """
        Synchronize local state with Twelve Data's actual
        credit headers.
        """

        credits_left_raw = response.headers.get(
            "api-credits-left"
        )

        credits_used_raw = response.headers.get(
            "api-credits-used"
        )

        try:
            credits_left = (
                int(credits_left_raw)
                if credits_left_raw is not None
                else None
            )
        except (TypeError, ValueError):
            credits_left = None

        try:
            credits_used = (
                int(credits_used_raw)
                if credits_used_raw is not None
                else None
            )
        except (TypeError, ValueError):
            credits_used = None

        if credits_left is not None:
            self._provider_credits_left = credits_left

        if credits_used is not None:
            self._provider_credits_used = credits_used

        if (
            credits_left is not None
            and credits_used is not None
        ):
            self._provider_limit = (
                credits_left + credits_used
            )

            # Never exceed the provider's real quota.
            #
            # Keep one credit as a safety reserve.
            self.max_credits_per_minute = max(
                1,
                self._provider_limit
                - self.safety_reserve,
            )

            logger.debug(
                "Twelve Data credits: "
                f"used={credits_used}, "
                f"left={credits_left}, "
                f"limit={self._provider_limit}, "
                f"local_limit={self.max_credits_per_minute}"
            )

    async def _block_until_reset(self) -> None:
        async with self._rate_lock:
            wait_seconds = (
                self._seconds_until_next_minute()
            )

            self._blocked_until = (
                time.monotonic() + wait_seconds
            )

            # The provider has told us the quota is exhausted.
            self._provider_credits_left = 0

            logger.warning(
                "Twelve Data quota exhausted. "
                f"Blocking requests for {wait_seconds:.1f}s "
                "until the next quota window."
            )

    async def _get(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> dict:
        """
        Perform one safe Twelve Data request.

        429:
            Never immediately retries.

        Network/timeout:
            Raised to the caller.

        5xx:
            Raised to the caller.

        API status=error:
            Converted to ValueError.
        """

        await self._wait_for_credit_slot(1)

        request_params = {
            **params,
            "apikey": self.api_key,
        }

        client = await self._get_client()

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

        # ---------------------------------------------------------
        # RATE LIMIT
        # ---------------------------------------------------------
        if response.status_code == 429:
            logger.warning(
                "Twelve Data returned HTTP 429 on "
                f"{endpoint}. "
                "No immediate retry will be performed."
            )

            await self._block_until_reset()

            raise TwelveDataRateLimitError(
                "Twelve Data API rate limit reached."
            )

        # ---------------------------------------------------------
        # SERVER ERRORS
        # ---------------------------------------------------------
        if 500 <= response.status_code <= 599:
            logger.warning(
                "Twelve Data server error "
                f"{response.status_code} on {endpoint}."
            )

            response.raise_for_status()

        # ---------------------------------------------------------
        # OTHER HTTP ERRORS
        # ---------------------------------------------------------
        response.raise_for_status()

        # ---------------------------------------------------------
        # JSON
        # ---------------------------------------------------------
        try:
            data = response.json()

        except ValueError as exc:
            logger.error(
                "Invalid JSON response from Twelve Data "
                f"endpoint={endpoint}"
            )

            raise ValueError(
                "Invalid JSON response from Twelve Data"
            ) from exc

        # ---------------------------------------------------------
        # API-LEVEL ERROR
        # ---------------------------------------------------------
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
        """Return latest quote for a pair."""

        return await self._get(
            "quote",
            {
                "symbol": self._symbol(pair),
            },
        )

    async def get_price(self, pair: str) -> float:
        """Return latest price for a pair."""

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
        Return OHLC data ordered oldest -> newest.

        Public signature remains unchanged.
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

        df = df.dropna(
            subset=["datetime"]
        )

        required_columns = [
            "open",
            "high",
            "low",
            "close",
        ]

        for column in required_columns:
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
            subset=required_columns
        )

        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(
                df["volume"],
                errors="coerce",
            ).fillna(0.0)
        else:
            df["volume"] = 0.0

        return (
            df
            .sort_values("datetime")
            .reset_index(drop=True)
        )

    async def get_quotes_bulk(
        self,
        pairs: list[str],
    ) -> dict:
        """
        Fetch quotes for multiple symbols.

        Note:
        Credit consumption is based on symbols and endpoint
        weight according to Twelve Data's credit model.
        """

        if not pairs:
            return {}

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
