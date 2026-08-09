# AI Forex Signal System — Backend (FastAPI + PostgreSQL + Firebase Auth)

## Authentication architecture

This backend uses **Firebase Authentication** as the identity provider —
there is no custom password storage in PostgreSQL. Two different Firebase
credentials are involved, and they are not interchangeable:

| Credential | Where to get it | Used for | Secrecy |
|---|---|---|---|
| Service account JSON | Firebase Console → Project settings → Service accounts → Generate new private key | Admin SDK: creating users, verifying ID tokens, sending FCM push | **Secret** — never expose, never commit |
| Web API key | Firebase Console → Project settings → General → "Web API Key" | REST sign-in (`/auth/login`) and token refresh (`/auth/refresh`) via Identity Toolkit | Same key embedded in any Firebase client app config — keep out of source control anyway, but it is not a high-value secret on its own |

**Why a REST call for login instead of just verifying tokens?** The Firebase
Admin SDK intentionally cannot check a password — that's a client-SDK-only
operation. Since this app authenticates via a plain `email` + `password` JSON
body (rather than having the Flutter app talk to Firebase directly), the
backend calls Firebase's own `accounts:signInWithPassword` REST endpoint to
validate the credentials server-side, then returns Firebase ID/refresh tokens
to the client.

## How the flow works

```
POST /api/v1/auth/register {full_name, email, password}
  → creates the user in Firebase Authentication
  → creates the matching profile row in PostgreSQL (users.firebase_uid)
  → signs the user in immediately (via Firebase REST) and returns tokens

POST /api/v1/auth/login {email, password}
  → validates the password against Firebase (REST)
  → looks up (or lazily creates) the PostgreSQL profile by firebase_uid
  → returns {user, id_token, refresh_token, expires_in}

Every other endpoint:
  → requires  Authorization: Bearer <id_token>
  → app.core.deps.get_current_user verifies the token with the Admin SDK
    (firebase_admin.auth.verify_id_token) and loads the matching profile

POST /api/v1/auth/refresh {refresh_token}
  → exchanges an expiring ID token for a new one via Firebase's
    securetoken.googleapis.com endpoint (same mechanism the Flutter app's
    Dio interceptor already calls on a 401)
```

## Setup

### 1. Firebase project

1. Create a project at console.firebase.google.com (or reuse the one from the Flutter app's push notification setup).
2. Enable **Authentication → Sign-in method → Email/Password**.
3. Project settings → Service accounts → **Generate new private key** → save the JSON.
4. Project settings → General → copy the **Web API Key**.

### 2. Environment variables

Copy `.env.example` to `.env` and fill in:

```bash
FIREBASE_CREDENTIALS_PATH=/path/to/firebase-service-account.json
FIREBASE_WEB_API_KEY=AIza...                 # from Project settings -> General
DATABASE_URL=postgresql://forex_user:...@localhost:5432/forex_signals
TWELVE_DATA_API_KEY=...
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=a-strong-password              # bootstrapped into Firebase + Postgres on first startup
```

### 3. Run it

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
uvicorn app.main:app --reload
```

On startup, the app creates the Postgres tables, initializes the Firebase
Admin SDK, and bootstraps the admin account (in both Firebase and Postgres)
if it doesn't already exist.

Or with Docker:

```bash
docker compose up --build
```

(Mount your service account JSON into the container at the path referenced
by `docker-compose.yml`'s volume — see the `firebase-service-account.json`
line in that file.)

## API reference (auth-relevant endpoints)

| Method | Path | Auth required | Body | Returns |
|---|---|---|---|---|
| POST | `/api/v1/auth/register` | No | `full_name, email, password` | `user`, `id_token`, `refresh_token`, `expires_in` |
| POST | `/api/v1/auth/login` | No | `email, password` | same shape as register |
| POST | `/api/v1/auth/refresh` | No | `refresh_token` | `id_token`, `refresh_token`, `expires_in` |
| GET | `/api/v1/auth/me` | Yes (Bearer ID token) | — | current user profile |

All other endpoints (`/signals`, `/prices`, `/users/*`, `/notifications`,
`/admin/*`) require `Authorization: Bearer <firebase_id_token>` exactly as
before — only the token *issuer* changed, not how the Flutter app attaches it
(the existing `ApiClient` interceptor in the Flutter app needs no changes to
how it sends the header, only to how it obtains the token — see the note
below).

## Flutter app follow-up required

The Flutter app built earlier calls `/auth/login` and `/auth/register` and
stores the returned `access_token`/`refresh_token` field names. Since this
backend now returns `id_token` instead of `access_token`, **the Flutter
app's `AuthService` and token-parsing code need a small update** to read
`id_token` instead of `access_token` (the rest of the app — `ApiClient`'s
Bearer header logic, secure storage, refresh-on-401 — needs no changes,
since it only cares about *a* token string, not its field name). Say the
word and I'll patch the Flutter app to match.

## Testing

```bash
pip install pytest pytest-mock --break-system-packages
pytest tests/ -v
```

- `test_indicators.py`, `test_patterns.py`, `test_smart_money.py` — pure-function
  unit tests for the AI signal engine, no external services needed.
- `test_firebase_auth.py` — mocked unit tests for the Firebase Auth service's
  error handling (duplicate email, expired token, missing config), no live
  Firebase project needed.
- Full endpoint integration testing (hitting `/auth/register` etc. against a
  real Postgres + real Firebase test project) needs live credentials this
  environment doesn't have; run it via `TestClient` against your own `.env`
  once deployed, or wire up a CI workflow with Postgres + Firebase test
  project secrets.

## Known limitation, stated plainly

I was unable to execute `pytest`, connect to a live Postgres instance, or
call the real Firebase REST endpoints while building this — the sandbox this
was built in has no network access. Every file has been manually
cross-checked (imports resolve, field names match Pydantic schemas, no
leftover references to the old JWT/password code), but the very first real
test is you running `uvicorn app.main:app` against your actual `.env`. If
something doesn't import cleanly, send me the traceback and I'll fix it
immediately.
