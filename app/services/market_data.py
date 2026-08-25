"""
Unified market-data client.

Primary provider:
    FCS API

Fallback provider:
    Twelve Data

FCS is used first.
If FCS fails for any reason, Twelve Data is used automatically.

API keys remain server-side.
The Flutter app never receives either API key.
"""

from typing import Any

import httpx
import pandas as pd
from loguru import logger

from app.config import settings
from app.services.twelve_data import (
    TwelveDataRateLimitError,
    twelve_data_client,
)


# =============================================================
# FCS SYMBOL MAP
# =============================================================

FCS_SYMBOL_MAP = {
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
    "USD/JPY": "USDJPY",
    "USD/CAD": "USDCAD",
    "AUD/USD": "AUDUSD",
    "NZD/USD": "NZDUSD",
    "EUR/JPY": "EURJPY",
    "GBP/JPY": "GBPJPY",
    "XAU/USD": "XAUUSD",
    "BTC/USD": "BTCUSD",
}


class FCSMarketDataError(Exception):
    """Raised when FCS cannot provide valid market data."""


class FCSClient:

    def __init__(self):
        self.base_url = (
            settings.FCS_BASE_URL.rstrip("/")
        )

        self.api_key = settings.FCS_API_KEY

    # =========================================================
    # SYMBOL
    # =========================================================

    def _symbol(self, pair: str) -> str:
        return FCS_SYMBOL_MAP.get(
            pair,
            pair.replace("/", ""),
        )

    # =========================================================
    # REQUEST
    # =========================================================

    async def _get(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> dict:

        if not self.api_key:

            raise FCSMarketDataError(
                "FCS_API_KEY is not configured."
            )

        request_params = {
            **params,
            "access_key": self.api_key,
        }

        url = (
            f"{self.base_url}/forex/"
            f"{endpoint.lstrip('/')}"
        )

        logger.debug(
            "FCS request: "
            f"endpoint={endpoint}, "
            f"params={params}"
        )

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
                    url,
                    params=request_params,
                )

            except httpx.TimeoutException as exc:

                logger.warning(
                    "FCS timeout on "
                    f"{endpoint}: {exc}"
                )

                raise FCSMarketDataError(
                    "FCS request timed out."
                ) from exc

            except httpx.NetworkError as exc:

                logger.warning(
                    "FCS network error on "
                    f"{endpoint}: {exc}"
                )

                raise FCSMarketDataError(
                    "FCS network error."
                ) from exc

            except Exception as exc:

                logger.warning(
                    "Unexpected FCS request error "
                    f"on {endpoint}: {exc}"
                )

                raise FCSMarketDataError(
                    "FCS request failed."
                ) from exc

            # -------------------------------------------------
            # HTTP STATUS
            # -------------------------------------------------

            if response.status_code >= 400:

                logger.warning(
                    "FCS HTTP error: "
                    f"status={response.status_code}, "
                    f"endpoint={endpoint}"
                )

                raise FCSMarketDataError(
                    f"FCS HTTP {response.status_code}"
                )

            # -------------------------------------------------
            # JSON
            # -------------------------------------------------

            try:

                data = response.json()

            except ValueError as exc:

                logger.warning(
                    "Invalid JSON response from FCS "
                    f"endpoint={endpoint}"
                )

                raise FCSMarketDataError(
                    "Invalid JSON response from FCS."
                ) from exc

        if not isinstance(data, dict):

            raise FCSMarketDataError(
                "Unexpected FCS response format."
            )

        # -----------------------------------------------------
        # FCS STATUS
        # -----------------------------------------------------

        status = data.get("status")

        if status is False:

            message = data.get(
                "msg",
                "FCS API returned an error.",
            )

            raise FCSMarketDataError(
                str(message)
            )

        # Some FCS responses may use string status.
        if isinstance(status, str):

            if status.lower() in {
                "false",
                "error",
                "failed",
            }:

                message = data.get(
                    "msg",
                    "FCS API returned an error.",
                )

                raise FCSMarketDataError(
                    str(message)
                )

        return data

    # =========================================================
    # HISTORY
    # =========================================================

    async def get_time_series(
        self,
        pair: str,
        interval: str = "1h",
        outputsize: int = 300,
    ) -> pd.DataFrame:

        symbol = self._symbol(pair)

        # -----------------------------------------------------
        # LIMIT OUTPUT SIZE
        # -----------------------------------------------------

        length = min(
            max(int(outputsize), 1),
            300,
        )

        # -----------------------------------------------------
        # FCS PERIOD
        # -----------------------------------------------------

        supported_periods = {
            "1m",
            "5m",
            "15m",
            "30m",
            "1h",
            "4h",
            "1d",
            "1w",
            "1mo",
        }

        period = interval

        if period not in supported_periods:

            raise FCSMarketDataError(
                f"Unsupported FCS interval: {interval}"
            )

        params: dict[str, Any] = {
            "symbol": symbol,
            "period": period,
            "length": length,
        }

        # -----------------------------------------------------
        # GOLD
        # -----------------------------------------------------
        #
        # FCS treats XAUUSD as a commodity.
        #

        if pair == "XAU/USD":

            params["type"] = "commodity"

        # -----------------------------------------------------
        # BTC
        # -----------------------------------------------------
        #
        # If FCS does not support BTC through this endpoint,
        # MarketDataClient will automatically fall back to
        # Twelve Data.
        #

        if pair == "BTC/USD":

            params["type"] = "crypto"

        # -----------------------------------------------------
        # REQUEST
        # -----------------------------------------------------

        data = await self._get(
            "history",
            params,
        )

        raw_response = data.get(
            "response"
        )

        if not raw_response:

            raise FCSMarketDataError(
                f"No FCS history returned for {pair}."
            )

        # -----------------------------------------------------
        # NORMALIZE RESPONSE
        # -----------------------------------------------------

        if isinstance(
            raw_response,
            list,
        ):

            rows = raw_response

        elif isinstance(
            raw_response,
            dict,
        ):

            # Common FCS response shape:
            #
            # response:
            # {
            #     "0": {...},
            #     "1": {...}
            # }
            #
            rows = list(
                raw_response.values()
            )

        else:

            raise FCSMarketDataError(
                f"Unexpected FCS history format "
                f"for {pair}."
            )

        rows = [
            row
            for row in rows
            if isinstance(row, dict)
        ]

        if not rows:

            raise FCSMarketDataError(
                f"FCS returned no candle rows "
                f"for {pair}."
            )

        df = pd.DataFrame(rows)

        # -----------------------------------------------------
        # OHLC COLUMN DETECTION
        # -----------------------------------------------------
        #
        # FCS normally returns:
        #
        # o = open
        # h = high
        # l = low
        # c = close
        #

        column_map = {
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }

        df = df.rename(
            columns=column_map
        )

        # -----------------------------------------------------
        # REQUIRED OHLC
        # -----------------------------------------------------

        for column in [
            "open",
            "high",
            "low",
            "close",
        ]:

            if column not in df.columns:

                raise FCSMarketDataError(
                    f"FCS response for {pair} "
                    f"is missing '{column}'."
                )

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        # -----------------------------------------------------
        # DATETIME
        # -----------------------------------------------------

        if "tm" in df.columns:

            df["datetime"] = pd.to_datetime(
                df["tm"],
                errors="coerce",
            )

        elif "t" in df.columns:

            timestamps = pd.to_numeric(
                df["t"],
                errors="coerce",
            )

            df["datetime"] = pd.to_datetime(
                timestamps,
                unit="s",
                errors="coerce",
            )

        elif "timestamp" in df.columns:

            timestamps = pd.to_numeric(
                df["timestamp"],
                errors="coerce",
            )

            df["datetime"] = pd.to_datetime(
                timestamps,
                unit="s",
                errors="coerce",
            )

        elif "date" in df.columns:

            df["datetime"] = pd.to_datetime(
                df["date"],
                errors="coerce",
            )

        else:

            raise FCSMarketDataError(
                f"FCS response for {pair} "
                "contains no recognized timestamp."
            )

        # -----------------------------------------------------
        # VOLUME
        # -----------------------------------------------------

        if "volume" in df.columns:

            df["volume"] = pd.to_numeric(
                df["volume"],
                errors="coerce",
            ).fillna(0.0)

        else:

            df["volume"] = 0.0

        # -----------------------------------------------------
        # CLEAN INVALID DATA
        # -----------------------------------------------------

        df = df.dropna(
            subset=[
                "datetime",
                "open",
                "high",
                "low",
                "close",
            ]
        )

        if df.empty:

            raise FCSMarketDataError(
                f"FCS returned invalid OHLC "
                f"data for {pair}."
            )

        # -----------------------------------------------------
        # SORT
        # -----------------------------------------------------

        df = (
            df.sort_values(
                "datetime"
            )
            .reset_index(
                drop=True
            )
        )

        # -----------------------------------------------------
        # FINAL FORMAT
        # -----------------------------------------------------

        final_columns = [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        df = df[
            [
                column
                for column in final_columns
                if column in df.columns
            ]
        ]

        logger.info(
            "FCS market data loaded: "
            f"{pair}, "
            f"interval={interval}, "
            f"candles={len(df)}"
        )

        return df


# =============================================================
# UNIFIED MARKET DATA CLIENT
# =============================================================

class MarketDataClient:
    """
    Unified market-data provider.

    FCS:
        Primary provider.

    Twelve Data:
        Automatic fallback.
    """

    def __init__(self):

        self.fcs = FCSClient()

    # =========================================================
    # TIME SERIES
    # =========================================================

    async def get_time_series(
        self,
        pair: str,
        interval: str = "1h",
        outputsize: int = 300,
    ) -> pd.DataFrame:

        # -----------------------------------------------------
        # PRIMARY: FCS
        # -----------------------------------------------------

        try:

            df = await self.fcs.get_time_series(
                pair=pair,
                interval=interval,
                outputsize=outputsize,
            )

            logger.success(
                f"[{pair}] Market data source: FCS"
            )

            return df

        except Exception as exc:

            logger.warning(
                f"[{pair}] FCS failed: {exc}. "
                "Trying Twelve Data fallback."
            )

        # -----------------------------------------------------
        # FALLBACK: TWELVE DATA
        # -----------------------------------------------------

        try:

            df = await twelve_data_client.get_time_series(
                pair=pair,
                interval=interval,
                outputsize=outputsize,
            )

            logger.success(
                f"[{pair}] Market data source: "
                "Twelve Data fallback"
            )

            return df

        except TwelveDataRateLimitError:

            logger.warning(
                f"[{pair}] FCS failed and "
                "Twelve Data fallback quota is exhausted."
            )

            raise

        except Exception as exc:

            logger.error(
                f"[{pair}] Both FCS and Twelve Data "
                f"failed: {exc}"
            )

            raise


# =============================================================
# SINGLETON
# =============================================================

market_data_client = MarketDataClient()
