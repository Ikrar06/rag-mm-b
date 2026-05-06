"""Auth service — simple JWT-based auth untuk prototype/demo.

User disimpan di users.json (plaintext password — DEMO ONLY, bukan production).
JWT disimpan di httpOnly cookie dan Authorization header.

Untuk POC production: ganti dengan proper bcrypt + database (lihat komentar di bawah).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import jwt, JWTError

# ─── Config ──────────────────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_USERS_FILE = os.path.join(_PROJECT_ROOT, "users.json")
_JWT_SECRET = os.getenv("JWT_SECRET", "unhas-rag-demo-secret-2026")
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))


# ─── User store ──────────────────────────────────────────────────────────────

def _load_users() -> list[dict]:
    """Load users from users.json."""
    if not os.path.exists(_USERS_FILE):
        return []
    with open(_USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Validate username + password. Return user dict or None."""
    users = _load_users()
    for user in users:
        if user["username"] == username and user["password"] == password:
            return user
    return None

    # POC production — replace above dengan bcrypt:
    # from passlib.context import CryptContext
    # pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
    # if user and pwd_ctx.verify(password, user["hashed_password"]):
    #     return user


# ─── JWT ─────────────────────────────────────────────────────────────────────

def create_token(user: dict) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=_JWT_EXPIRE_HOURS)
    payload = {
        "sub": user["username"],
        "role": user["role"],
        "name": user.get("name", user["username"]),
        "nim": user.get("nim"),
        "exp": expire,
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Decode JWT. Return payload dict or None if invalid/expired."""
    try:
        return jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
    except JWTError:
        return None


def get_role_access_levels(role: str) -> list[str]:
    """Mapping role → access levels dokumen yang boleh diakses."""
    if role == "calon_mahasiswa":
        return ["public"]
    if role == "mahasiswa":
        return ["public", "mahasiswa"]
    if role == "staf_akademik":
        return ["public", "mahasiswa", "staf_akademik"]
    return ["public"]
