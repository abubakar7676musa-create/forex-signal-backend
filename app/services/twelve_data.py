"""
Secure Twelve Data API client.

Goals:
- Never expose the API key to the Flutter app.
- Prevent HTTP 429 retry storms.
- Respect Twelve Data minute-based API credit limits.
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
        self.base_url = (
            settings.TWELVE_DATA_BASE_URL.rstrip("/")
        )

        self.api_key = settings.TWELVE_DATA_API_KEY

        # Twelve Data Basic/Free:
        # 8 API credits per minute.
        #
        # Reserve one credit as a safety margin.
        self.max_credits_per_minute = 7

        # Local reservations.
        # (monotonic timestamp, estimated credits)
        self._credit_history: deque[
            tuple[float, int]
        ] = deque()

        # Provider-reported remaining credits.
        self._provider_credits_left: int | None = None

        # Protect limiter state.
        self._rate_lock = asyncio.Lock()

        # If provider reports quota exhaustion,
        # block until the next minute.
        self._blocked_until: float = 0.0

    def _symbol(self, pair: str) -> str:
        return SYMBOL_MAP.get(pair, pair)

    def _seconds_until_next_minute(self) -> float:
        """
        Twelve Data resets minute quota at the start
        of the next minute.

        Add a small safety margin.
        """

        now = time.time()

        next_minute = (
            int(now // 60) + 1
        ) * 60

        return max(
            1.0,
            next_minute - now + 0.5,
        )

    def _cleanup_credit_history(self) -> None:
        """
        Remove reservations older than one minute.
        """

        now = time.monotonic()

        while self._credit_history:

            timestamp, _credits = (
                self._credit_history[0]
            )

            if now - timestamp >= 60:
                self._credit_history.popleft()
            else:
                break

    def _local_credits_used(self) -> int:
        return sum(
            credits
            for _timestamp, credits
            in self._credit_history
        )

    async def _wait_for_credit_slot(
        self,
        estimated_credits: int = 1,
    ) -> None:
        """
        Wait until both local and provider-reported
        credit limits allow a request.
        """

        while True:

            async with self._rate_lock:

                now = time.monotonic()

                # ---------------------------------------------
                # PROVIDER BLOCK
                # ---------------------------------------------

                if now < self._blocked_until:

                    wait_seconds = (
                        self._blocked_until - now
                    )

                    logger.warning(
                        "Twelve Data quota exhausted. "
                        f"Waiting {wait_seconds:.1f}s "
                        "for the next quota window."
                    )

                else:

                    self._cleanup_credit_history()

                    local_used = (
                        self._local_credits_used()
                    )

                    # -----------------------------------------
                    # PROVIDER-REPORTED LIMIT
                    # -----------------------------------------

                    provider_left = (
                        self._provider_credits_left
                    )

                    provider_allows = (
                        provider_left is None
                        or provider_left
                        >= estimated_credits
                    )

                    local_allows = (
                        local_used
                        + estimated_credits
                        <= self.max_credits_per_minute
                    )

                    if (
                        local_allows
                        and provider_allows
                    ):
                        self._credit_history.append(
                            (
                                time.monotonic(),
                                estimated_credits,
                            )
                        )

                        logger.debug(
                            "Twelve Data credit slot "
                            f"reserved: "
                            f"local_used={local_used}, "
                            f"local_limit="
                            f"{self.max_credits_per_minute}, "
                            f"provider_left="
                            f"{provider_left}"
                        )

                        return

                    # -----------------------------------------
                    # WAIT FOR NEXT MINUTE
                    # -----------------------------------------

                    wait_seconds = (
                        self._seconds_until_next_minute()
                    )

                    logger.info(
                        "Twelve Data rate limiter waiting. "
                        f"local_used={local_used}/"
                        f"{self.max_credits_per_minute}, "
                        f"provider_left="
                        f"{provider_left}, "
                        f"waiting="
                        f"{wait_seconds:.1f}s"
                    )

            await asyncio.sleep(
                min(wait_seconds, 5.0)
            )

    def _remove_last_reservation(self) -> None:
        """
        Remove the most recent local reservation.

        Used when the provider rejected the request.
        """

        self._cleanup_credit_history()

        if not self._credit_history:
            return

        timestamp, credits = (
            self._credit_history[-1]
        )

        if (
            time.monotonic() - timestamp < 5
            and credits == 1
        ):
            self._credit_history.pop()

    def _sync_credit_headers(
        self,
        response: httpx.Response,
    ) -> None:
        """
        Synchronize local state with Twelve Data
        provider credit headers.
        """

        credits_left = response.headers.get(
            "api-credits-left"
        )

        credits_used = response.headers.get(
            "api-credits-used"
        )

        credits_request = response.headers.get(
            "api-credits-request"
        )

        try:
            if credits_left is not None:
                self._provider_credits_left = int(
                    credits_left
                )

        except ValueError:
            logger.warning(
                f"Invalid api-credits-left header: "
                f"{credits_left}"
            )

        logger.debug(
            "Twelve Data credits: "
            f"used={credits_used}, "
            f"left={credits_left}, "
            f"request={credits_request}, "
            f"local_limit="
            f"{self.max_credits_per_minute}"
        )

        # If provider says no credits remain,
        # proactively block requests until the next minute.
        try:
            if (
                credits_left is not None
                and int(credits_left) <= 0
            ):
                self._blocked_until = (
                    time.monotonic()
                    + self._seconds_until_next_minute()
                )

        except ValueError:
            pass

    async def _get(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> dict:

        # Current endpoints used by the application
        # consume one credit per symbol/request.
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
                    f"Twelve Data timeout on "
                    f"{endpoint}: {exc}"
                )

                # Timeout does not prove whether the provider
                # consumed the credit. Keep reservation.
                raise

            except httpx.NetworkError as exc:

                logger.warning(
                    f"Twelve Data network error on "
                    f"{endpoint}: {exc}"
                )

                # Network failure does not prove whether
                # provider consumed the request.
                raise

            self._sync_credit_headers(
                response
            )

            # -------------------------------------------------
            # RATE LIMIT
            # -------------------------------------------------

            if response.status_code == 429:

                async with self._rate_lock:

                    self._remove_last_reservation()

                    self._blocked_until = (
                        time.monotonic()
                        + self._seconds_until_next_minute()
                    )

                logger.warning(
                    "Twelve Data returned HTTP 429. "
                    "No immediate retry will be performed. "
                    "Requests are blocked until the next "
                    "quota window."
                )

                raise TwelveDataRateLimitError(
                    "Twelve Data API rate limit reached."
                )

            # -------------------------------------------------
            # SERVER ERRORS
            # -------------------------------------------------

            if (
                500
                <= response.status_code
                <= 599
            ):

                logger.warning(
                    "Twelve Data server error "
                    f"{response.status_code} "
                    f"on {endpoint}."
                )

                response.raise_for_status()

            # -------------------------------------------------
            # OTHER HTTP ERRORS
            # -------------------------------------------------

            response.raise_for_status()

            # -------------------------------------------------
            # JSON
            # -------------------------------------------------

            try:

                data = response.json()

            except ValueError as exc:

                logger.error(
                    "Invalid JSON response from "
                    f"Twelve Data endpoint="
                    f"{endpoint}"
                )

                raise ValueError(
                    "Invalid JSON response from Twelve Data"
                ) from exc

            # -------------------------------------------------
            # API-LEVEL ERROR
            # -------------------------------------------------

            if isinstance(data, dict):

                if data.get("status") == "error":

                    message = data.get(
                        "message",
                        "Twelve Data API error",
                    )

                    logger.error(
                        f"Twelve Data API error "
                        f"on {endpoint}: "
                        f"{message}"
                    )

                    raise ValueError(message)

            return data

    async def get_quote(
        self,
        pair: str,
    ) -> dict:

        return await self._get(
            "quote",
            {
                "symbol": self._symbol(pair),
            },
        )

    async def get_price(
        self,
        pair: str,
    ) -> float:

        data = await self._get(
            "price",
            {
                "symbol": self._symbol(pair),
            },
        )

        return float(
            data["price"]
        )

    async def get_time_series(
        self,
        pair: str,
        interval: str = "1h",
        outputsize: int = 300,
    ) -> pd.DataFrame:

        data = await self._get(
            "time_series",
            {
                "symbol": self._symbol(pair),
                "interval": interval,
                "outputsize": outputsize,
            },
        )

        values = data.get(
            "values",
            [],
        )

        if not values:
            raise ValueError(
                f"No time series data returned "
                f"for {pair}"
            )

        df = pd.DataFrame(
            values
        )

        if "datetime" not in df.columns:

            raise ValueError(
                f"Twelve Data response for "
                f"{pair} does not contain "
                "datetime values."
            )

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
        )

        df = df.dropna(
            subset=["datetime"]
        )

        for column in [
            "open",
            "high",
            "low",
            "close",
        ]:

            if column not in df.columns:

                raise ValueError(
                    f"Missing required column "
                    f"'{column}' for {pair}"
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
            df.sort_values(
                "datetime"
            )
            .reset_index(
                drop=True
            )
        )

    async def get_quotes_bulk(
        self,
        pairs: list[str],
    ) -> dict:

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
