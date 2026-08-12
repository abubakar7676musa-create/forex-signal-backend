import time
import uuid

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config import settings
from app.database import Base, engine, SessionLocal
from app.models.user import User, UserRole
from app.core.firebase import init_firebase
from app.services.firebase_auth import create_firebase_user
from firebase_admin import auth as firebase_auth
from app.services.scheduler import start_scheduler
from app.api import auth, signals, prices, users, admin, notifications

# --- Rate limiting ---
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.RATE_LIMIT_PER_MINUTE}/minute"])

app = FastAPI(
    title="AI Forex Signal System API",
    description="Production backend serving AI-generated Forex/Gold/BTC Buy/Sell signals.",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


# --- Render health check ---
# Render's health check hits GET /healthz specifically (not /health, which is
# also kept below for any existing monitors already pointing at it). This
# handler is intentionally trivial: no DB session, no external API calls, no
# auth — so it returns 200 even if Postgres, Twelve Data, or Firebase are
# temporarily unreachable. It's also exempt from the SlowAPI rate limiter so
# frequent health-check polling can never itself cause a failed check.
@app.get("/healthz", tags=["health"])
@limiter.exempt
def healthz(request: Request):
    return {"status": "ok"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_and_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.time()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(f"[{request_id}] Unhandled error on {request.method} {request.url.path}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "request_id": request_id},
        )
    duration_ms = round((time.time() - start) * 1000, 1)
    logger.info(f"[{request_id}] {request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)")
    response.headers["X-Request-ID"] = request_id
    return response


# --- Routers ---
app.include_router(auth.router)
app.include_router(signals.router)
app.include_router(prices.router)
app.include_router(users.router)
app.include_router(notifications.router)
app.include_router(admin.router)


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok", "environment": settings.ENVIRONMENT}


def _ensure_admin_user():
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == settings.ADMIN_EMAIL.lower()).first()
        if existing:
            return

        # Look up (or create) the Firebase-side account first.
        try:
            firebase_user = firebase_auth.get_user_by_email(settings.ADMIN_EMAIL.lower())
        except firebase_auth.UserNotFoundError:
            firebase_user = create_firebase_user(
                email=settings.ADMIN_EMAIL.lower(),
                password=settings.ADMIN_PASSWORD,
                full_name="Administrator",
            )

        admin_user = User(
            firebase_uid=firebase_user.uid,
            email=settings.ADMIN_EMAIL.lower(),
            full_name="Administrator",
            role=UserRole.admin,
            favorite_pairs=[],
        )
        db.add(admin_user)
        db.commit()
        logger.info(f"Bootstrapped admin user: {settings.ADMIN_EMAIL}")
    except Exception as e:
        logger.warning(f"Admin bootstrap skipped (Firebase not configured yet?): {e}")
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    init_firebase()
    _ensure_admin_user()
    start_scheduler()
    logger.info(f"AI Forex Signal System API started in '{settings.ENVIRONMENT}' mode.")
