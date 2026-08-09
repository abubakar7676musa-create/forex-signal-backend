import uuid
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr


class UserOut(BaseModel):
    id: uuid.UUID
    email: EmailStr
    full_name: Optional[str]
    role: str
    is_active: bool
    favorite_pairs: List[str] = []
    created_at: datetime

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    favorite_pairs: Optional[List[str]] = None


class FcmTokenUpdate(BaseModel):
    fcm_token: str
