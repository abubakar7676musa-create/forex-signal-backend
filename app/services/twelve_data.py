"""
Secure wrapper around the Twelve Data API.
The API key is read only from the server-side environment (TWELVE_DATA_API_KEY)
and is NEVER sent to, or accessible from, the Flutter app.
"""
import httpx
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

# Twelve Data uses no-slash symbols for FX/metals/crypto, e.g. EUR/USD -> EUR/USD works too,
# but we normalize here in case a pair needs remapping (e.g. Gold, BTC).
SYMBOL_MAP = {
    "XAU/USD": "XAU/USD",
    "BTC/USD": "BTC/USD",
}


class TwelveDataClient:
    def __init__(self):
        self.base_url = settings.TWELVE_DATA_BASE_URL
        self.api_key = settings.TWELVE_DATA_API_KEY

    def _symbol(self, pair: str) -> str:
        return SYMBOL_MAP.get(pair, pair)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _get(self, endpoint: str, params: dict) -> dict:
        params = {**params, "apikey": self.api_key}
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{self.base_url}/{endpoint}", params=params)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("status") == "error":
                logger.error(f"Twelve Data error for {endpoint} {params.get('symbol')}: {data.get('message')}")
                raise ValueError(data.get("message", "Twelve Data API error"))
            return data

    async def get_quote(self, pair: str) -> dict:
        """Latest quote: price, change, percent_change, etc."""
        return await self._get("quote", {"symbol": self._symbol(pair)})

    async def get_price(self, pair: str) -> float:
        data = await self._get("price", {"symbol": self._symbol(pair)})
        return float(data["price"])

    async def get_time_series(self, pair: str, interval: str = "1h", outputsize: int = 300) -> pd.DataFrame:
        """
        Returns a DataFrame indexed oldest->newest with columns:
        open, high, low, close, volume (volume may be NaN for FX pairs — Twelve Data
        does not provide true volume for most FX; we treat it as tick-volume proxy).
        """
        data = await self._get(
            "time_series",
            {"symbol": self._symbol(pair), "interval": interval, "outputsize": outputsize},
        )
        values = data.get("values", [])
        if not values:
            raise ValueError(f"No time series data returned for {pair}")

        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["volume"] = df.get("volume", pd.Series([0] * len(df))).astype(float)
        df = df.sort_values("datetime").reset_index(drop=True)
        return df

    async def get_quotes_bulk(self, pairs: list[str]) -> dict:
        """Fetch quotes for multiple pairs in one call (Twelve Data supports comma-separated symbols)."""
        symbols = ",".join(self._symbol(p) for p in pairs)
        data = await self._get("quote", {"symbol": symbols})
        # When multiple symbols are requested, Twelve Data returns a dict keyed by symbol
        if "symbol" in data:
            return {data["symbol"]: data}
        return data


twelve_data_client = TwelveDataClient()
