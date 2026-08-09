from pydantic import BaseModel, EmailStr, Field

from app.schemas.user import UserOut


class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=1, max_length=255)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)  # Firebase minimum is 6 characters


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class FirebaseTokenResponse(BaseModel):
    id_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class AuthResponse(BaseModel):
    """Returned by /auth/register and /auth/login: the Postgres profile plus
    Firebase tokens the client needs to authenticate subsequent requests."""
    user: UserOut
    id_token: str
    refresh_token: str
    expires_in: int

