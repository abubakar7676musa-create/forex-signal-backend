"""
Secure Twelve Data API client.

Goals:
- Never expose the API key to the Flutter app.
- NEVER wait for credit slots.
- NEVER create retry/wait storms.
- Stop immediately when Twelve Data quota is exhausted.
- Respect local minute credit limit.
- Read Twelve Data credit information from response headers.
- Keep existing public methods compatible with the app.
"""

from collections import deque
from typing import Any
import time

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

        # ---------------------------------------------------------
        # TWELVE DATA LOCAL LIMIT
        # ---------------------------------------------------------
        #
        # Keep one credit as safety margin.
        #
        # IMPORTANT:
        # We DO NOT WAIT when this limit is reached.
        # We immediately raise TwelveDataRateLimitError.
        #
        self.max_credits_per_minute = 7

        # (timestamp, estimated credits)
        self._credit_history: deque[
            tuple[float, int]
        ] = deque()

        # Provider-reported remaining credits.
        self._provider_credits_left: int | None = None

        # Provider block timestamp.
        self._blocked_until: float = 0.0

    # =============================================================
    # SYMBOL
    # =============================================================

    def _symbol(self, pair: str) -> str:
        return SYMBOL_MAP.get(pair, pair)

    # =============================================================
    # NEXT MINUTE
    # =============================================================

    def _seconds_until_next_minute(self) -> float:
        now = time.time()

        next_minute = (
            int(now // 60) + 1
        ) * 60

        return max(
            1.0,
            next_minute - now + 0.5,
        )

    # =============================================================
    # CLEAN OLD LOCAL CREDITS
    # =============================================================

    def _cleanup_credit_history(self) -> None:
        now = time.monotonic()

        while self._credit_history:

            timestamp, _credits = (
                self._credit_history[0]
            )

            if now - timestamp >= 60:
                self._credit_history.popleft()

            else:
                break

    # =============================================================
    # LOCAL CREDITS USED
    # =============================================================

    def _local_credits_used(self) -> int:
        return sum(
            credits
            for _timestamp, credits
            in self._credit_history
        )

    # =============================================================
    # CREDIT CHECK
    # =============================================================

    def _check_credit_slot(
        self,
        estimated_credits: int = 1,
    ) -> None:
        """
        Check whether a request is allowed.

        IMPORTANT:
        This function NEVER waits.

        If quota is unavailable:
            raise TwelveDataRateLimitError

        This completely prevents:
            waiting=30s
            waiting=25s
            waiting=20s
            etc.
        """

        now = time.monotonic()

        # ---------------------------------------------------------
        # PROVIDER TEMPORARY BLOCK
        # ---------------------------------------------------------

        if now < self._blocked_until:

            wait_seconds = (
                self._blocked_until - now
            )

            logger.warning(
                "Twelve Data quota is currently blocked. "
                f"Next quota window in "
                f"{wait_seconds:.1f}s. "
                "Request cancelled immediately."
            )

            raise TwelveDataRateLimitError(
                "Twelve Data quota temporarily exhausted."
            )

        # ---------------------------------------------------------
        # CLEAN LOCAL HISTORY
        # ---------------------------------------------------------

        self._cleanup_credit_history()

        local_used = (
            self._local_credits_used()
        )

        # ---------------------------------------------------------
        # PROVIDER-REPORTED CREDITS
        # ---------------------------------------------------------

        provider_left = (
            self._provider_credits_left
        )

        if (
            provider_left is not None
            and provider_left < estimated_credits
        ):

            wait_seconds = (
                self._seconds_until_next_minute()
            )

            self._blocked_until = (
                time.monotonic()
                + wait_seconds
            )

            logger.warning(
                "Twelve Data provider quota exhausted. "
                f"provider_left={provider_left}. "
                f"Next quota window in "
                f"{wait_seconds:.1f}s. "
                "Request cancelled immediately."
            )

            raise TwelveDataRateLimitError(
                "Twelve Data provider quota exhausted."
            )

        # ---------------------------------------------------------
        # LOCAL LIMIT
        # ---------------------------------------------------------

        if (
            local_used + estimated_credits
            > self.max_credits_per_minute
        ):

            wait_seconds = (
                self._seconds_until_next_minute()
            )

            self._blocked_until = (
                time.monotonic()
                + wait_seconds
            )

            logger.warning(
                "Local Twelve Data credit limit reached. "
                f"used={local_used}/"
                f"{self.max_credits_per_minute}. "
                f"Next quota window in "
                f"{wait_seconds:.1f}s. "
                "Request cancelled immediately."
            )

            raise TwelveDataRateLimitError(
                "Local Twelve Data credit limit reached."
            )

        # ---------------------------------------------------------
        # RESERVE CREDIT
        # ---------------------------------------------------------

        self._credit_history.append(
            (
                time.monotonic(),
                estimated_credits,
            )
        )

        logger.debug(
            "Twelve Data credit reserved: "
            f"used={local_used + estimated_credits}/"
            f"{self.max_credits_per_minute}, "
            f"provider_left={provider_left}"
        )

    # =============================================================
    # REMOVE LAST RESERVATION
    # =============================================================

    def _remove_last_reservation(self) -> None:
        """
        Remove the most recent local reservation.

        Used when provider returns HTTP 429.
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

    # =============================================================
    # SYNC PROVIDER HEADERS
    # =============================================================

    def _sync_credit_headers(
        self,
        response: httpx.Response,
    ) -> None:

        credits_left = response.headers.get(
            "api-credits-left"
        )

        credits_used = response.headers.get(
            "api-credits-used"
        )

        credits_request = response.headers.get(
            "api-credits-request"
        )

        # ---------------------------------------------------------
        # SAVE PROVIDER REMAINING CREDITS
        # ---------------------------------------------------------

        try:

            if credits_left is not None:

                self._provider_credits_left = int(
                    credits_left
                )

        except (ValueError, TypeError):

            logger.warning(
                "Invalid api-credits-left header: "
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

        # ---------------------------------------------------------
        # PROVIDER QUOTA EXHAUSTED
        # ---------------------------------------------------------

        try:

            if (
                credits_left is not None
                and int(credits_left) <= 0
            ):

                wait_seconds = (
                    self._seconds_until_next_minute()
                )

                self._blocked_until = (
                    time.monotonic()
                    + wait_seconds
                )

                logger.warning(
                    "Twelve Data credits exhausted. "
                    f"Next quota window in "
                    f"{wait_seconds:.1f}s."
                )

        except (ValueError, TypeError):
            pass

    # =============================================================
    # GET REQUEST
    # =============================================================

    async def _get(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> dict:

        # ---------------------------------------------------------
        # CHECK CREDIT
        # ---------------------------------------------------------
        #
        # IMPORTANT:
        # NO WAITING.
        #
        self._check_credit_slot(1)

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
                    "Twelve Data timeout on "
                    f"{endpoint}: {exc}"
                )

                # Do NOT retry.
                raise

            except httpx.NetworkError as exc:

                logger.warning(
                    "Twelve Data network error on "
                    f"{endpoint}: {exc}"
                )

                # Do NOT retry.
                raise

            # -----------------------------------------------------
            # READ CREDIT HEADERS
            # -----------------------------------------------------

            self._sync_credit_headers(
                response
            )

            # -----------------------------------------------------
            # HTTP 429
            # -----------------------------------------------------

            if response.status_code == 429:

                self._remove_last_reservation()

                wait_seconds = (
                    self._seconds_until_next_minute()
                )

                self._blocked_until = (
                    time.monotonic()
                    + wait_seconds
                )

                logger.warning(
                    "Twelve Data returned HTTP 429. "
                    f"Next quota window in "
                    f"{wait_seconds:.1f}s. "
                    "NO RETRY. "
                    "Request cancelled."
                )

                raise TwelveDataRateLimitError(
                    "Twelve Data API rate limit reached."
                )

            # -----------------------------------------------------
            # SERVER ERRORS
            # -----------------------------------------------------

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

            # -----------------------------------------------------
            # OTHER HTTP ERRORS
            # -----------------------------------------------------

            response.raise_for_status()

            # -----------------------------------------------------
            # JSON
            # -----------------------------------------------------

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

            # -----------------------------------------------------
            # API LEVEL ERROR
            # -----------------------------------------------------

            if isinstance(data, dict):

                if data.get("status") == "error":

                    message = data.get(
                        "message",
                        "Twelve Data API error",
                    )

                    logger.error(
                        "Twelve Data API error "
                        f"on {endpoint}: "
                        f"{message}"
                    )

                    raise ValueError(
                        message
                    )

            return data

    # =============================================================
    # QUOTE
    # =============================================================

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

    # =============================================================
    # PRICE
    # =============================================================

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

    # =============================================================
    # TIME SERIES
    # =============================================================

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

        # ---------------------------------------------------------
        # OHLC
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # VOLUME
        # ---------------------------------------------------------

        if "volume" in df.columns:

            df["volume"] = pd.to_numeric(
                df["volume"],
                errors="coerce",
            ).fillna(0.0)

        else:

            df["volume"] = 0.0

        # ---------------------------------------------------------
        # SORT
        # ---------------------------------------------------------

        return (
            df.sort_values(
                "datetime"
            )
            .reset_index(
                drop=True
            )
        )

    # =============================================================
    # BULK QUOTES
    # =============================================================

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


# =============================================================
# SINGLETON
# =============================================================

twelve_data_client = TwelveDataClient()
