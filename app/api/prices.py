from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app.config import SUPPORTED_PAIRS
from app.models.user import User
from app.schemas.signal import PriceOut
from app.core.deps import get_current_user
from app.services.twelve_data import twelve_data_client

router = APIRouter(prefix="/api/v1/prices", tags=["prices"])


@router.get("", response_model=list[PriceOut])
async def get_all_prices(current_user: User = Depends(get_current_user)):
    """Live quote for every supported pair, used for the dashboard price ticker."""
    results = []
    try:
        quotes = await twelve_data_client.get_quotes_bulk(SUPPORTED_PAIRS)
    except Exception as e:
        logger.error(f"Bulk quote fetch failed: {e}")
        raise HTTPException(status_code=502, detail="Live price feed temporarily unavailable")

    for pair in SUPPORTED_PAIRS:
        q = quotes.get(pair) or quotes.get(pair.replace("/", ""))
        if not q:
            continue
        try:
            results.append(PriceOut(
                pair=pair,
                price=float(q["close"]),
                change_percent=float(q.get("percent_change", 0)),
                timestamp=datetime.utcnow(),
            ))
        except (KeyError, ValueError, TypeError):
            continue
    return results


@router.get("/{pair:path}", response_model=PriceOut)
async def get_price(pair: str, current_user: User = Depends(get_current_user)):
    if pair not in SUPPORTED_PAIRS:
        raise HTTPException(status_code=400, detail=f"Unsupported pair: {pair}")
    try:
        quote = await twelve_data_client.get_quote(pair)
        return PriceOut(
            pair=pair,
            price=float(quote["close"]),
            change_percent=float(quote.get("percent_change", 0)),
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Quote fetch failed for {pair}: {e}")
        raise HTTPException(status_code=502, detail="Live price feed temporarily unavailable")
