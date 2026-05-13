"""Auth service — JWT-based auth dengan bcrypt + PostgreSQL.

Primary: DB User model dengan bcrypt password hash.
Fallback: users.json (dev/demo — plaintext password).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import jwt, JWTError

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_USERS_FILE = os.path.join(_PROJECT_ROOT, "users.json")
_JWT_SECRET = os.getenv("JWT_SECRET", "unhas-rag-demo-secret-2026")
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))


# ─── Password helpers ─────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ─── User store ──────────────────────────────────────────────────────────────

def _load_users_json() -> list[dict]:
    if not os.path.exists(_USERS_FILE):
        return []
    try:
        with open(_USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Validate credentials. Try DB first, fall back to users.json."""
    # ── Primary: PostgreSQL + bcrypt ─────────────────────────────────────────
    try:
        from backend.db.database import SessionLocal
        from backend.db.models import User

        db = SessionLocal()
        try:
            user = db.query(User).filter(
                User.username == username,
                User.is_active.is_(True),
            ).first()
            if user and verify_password(password, user.password_hash):
                return {
                    "username": user.username,
                    "role": user.role,
                    "name": user.name,
                    "nim": user.nim,
                }
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"auth_db_unavailable falling_back_to_json reason={e}")

    # ── Fallback: users.json (dev/demo only) ─────────────────────────────────
    for u in _load_users_json():
        if u.get("username") == username and u.get("password") == password:
            logger.warning(f"auth_via_json username={username} (plaintext — dev only)")
            return {
                "username": u["username"],
                "role": u.get("role", "public"),
                "name": u.get("name", username),
                "nim": u.get("nim"),
            }

    return None


def seed_users_from_json() -> None:
    """Seed users dari users.json ke DB dengan bcrypt hash.
    Hanya berjalan jika tabel users kosong (first-time setup).
    """
    users_data = _load_users_json()
    if not users_data:
        return

    try:
        from backend.db.database import SessionLocal
        from backend.db.models import User

        db = SessionLocal()
        try:
            if db.query(User).count() > 0:
                return  # sudah ada users, skip

            for u in users_data:
                plain_password = u.get("password", "")
                if not plain_password:
                    continue
                db.add(User(
                    username=u["username"],
                    password_hash=hash_password(plain_password),
                    name=u.get("name", u["username"]),
                    nim=u.get("nim"),
                    role=u.get("role", "public"),
                    is_active=True,
                ))
            db.commit()
            logger.info(f"users_seeded count={len(users_data)}")
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"users_seed_failed reason={e}")


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
    """Decode JWT. Return payload dict atau None jika invalid/expired."""
    try:
        return jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
    except JWTError:
        return None


def get_role_access_levels(role: str) -> list[str]:
    if role == "calon_mahasiswa":
        return ["public"]
    if role == "mahasiswa":
        return ["public", "mahasiswa"]
    if role == "staf_akademik":
        return ["public", "mahasiswa", "staf_akademik"]
    return ["public"]
