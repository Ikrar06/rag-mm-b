"""SQLAlchemy engine + session factory.

Sync SQLAlchemy (psycopg2) — simple untuk prototype.
Upgrade ke async (asyncpg) saat perlu scale di POC.
"""

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from backend.config import DATABASE_URL

logger = logging.getLogger(__name__)

# SQLAlchemy engine (sync)
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       # reconnect jika koneksi putus
    pool_size=20,
    max_overflow=20,
    echo=False,               # set True untuk debug SQL queries
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency untuk mendapatkan DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Buat semua tabel kalau belum ada + apply column additions yang idempotent.

    Dipanggil saat startup. Untuk schema change yang lebih kompleks (rename, drop,
    foreign key), pakai Alembic migration eksplisit.
    """
    from backend.db.models import User, Session, Message  # noqa: F401 — ensure tables are registered
    from sqlalchemy import text
    try:
        Base.metadata.create_all(bind=engine)

        # Migration ringan: ADD COLUMN IF NOT EXISTS (PostgreSQL 9.6+).
        # Aman untuk re-run — tidak error kalau column sudah ada.
        migrations = [
            "ALTER TABLE messages ADD COLUMN IF NOT EXISTS images JSON",
        ]
        with engine.begin() as conn:
            for ddl in migrations:
                conn.execute(text(ddl))

        logger.info("Database tables initialized + migrations applied.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def check_db_connection() -> bool:
    """Cek apakah PostgreSQL bisa diakses."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
