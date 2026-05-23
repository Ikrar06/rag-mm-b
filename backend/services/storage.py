"""Storage abstraction untuk gambar user (vision attachments).

Dev:  STORAGE_BACKEND=filesystem  (simpan di ./storage/)
POC:  STORAGE_BACKEND=minio       (MinIO container, akses via presigned URL)

API:
    storage = get_storage()
    url = storage.put_sync(key, bytes, content_type)
    # url adalah URL yang bisa di-share / di-buka di browser (presigned untuk MinIO).

Synchronous API dipilih supaya kompatibel dengan rag_pipeline yang masih sync.
"""

from __future__ import annotations

import io
import logging
import os
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class Storage(ABC):
    @abstractmethod
    def put_sync(self, key: str, data: bytes, content_type: str) -> str:
        """Simpan data, return URL yang bisa di-share/di-buka."""

    @abstractmethod
    def get_sync(self, key: str) -> bytes: ...

    @abstractmethod
    def delete_sync(self, key: str) -> None: ...


# =============================================================================
# Dev: filesystem storage
# =============================================================================

class FilesystemStorage(Storage):
    """Simpan ke disk lokal, expose via endpoint /api/files/{key}.

    Hanya untuk dev. Tidak akan accessible dari QA Sheet eksternal kecuali
    backend di-expose ke publik.
    """

    def __init__(self, base: str):
        self.base = Path(base)
        self.base.mkdir(parents=True, exist_ok=True)
        # Public URL prefix — untuk dev biasanya http://localhost:8000
        self.public_base = os.getenv("BACKEND_PUBLIC_URL", "http://localhost:8000").rstrip("/")

    def put_sync(self, key: str, data: bytes, content_type: str) -> str:
        path = self.base / key
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return f"{self.public_base}/api/files/{key}"

    def get_sync(self, key: str) -> bytes:
        with open(self.base / key, "rb") as f:
            return f.read()

    def delete_sync(self, key: str) -> None:
        path = self.base / key
        if path.exists():
            path.unlink()


# =============================================================================
# POC: MinIO storage dengan presigned URL
# =============================================================================

class MinIOStorage(Storage):
    """Upload ke MinIO + return presigned URL.

    Presigned URL valid 7 hari — cukup untuk window QA evaluation, tidak butuh
    public-read bucket (lebih aman). Reviewer klik URL di Sheet → buka di browser.
    """

    PRESIGNED_EXPIRY = timedelta(days=7)

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
        public_endpoint: str | None = None,
    ):
        from minio import Minio

        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.bucket = bucket
        # Endpoint yang user/QA reviewer pakai untuk download (dari browser).
        # Default ke endpoint internal — di POC sebaiknya di-override ke domain publik.
        self.public_endpoint = public_endpoint or endpoint
        self.secure = secure

        # Ensure bucket ada.
        try:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                logger.info("minio_bucket_created bucket=%s", bucket)
        except Exception as e:
            logger.error("minio_bucket_check_failed bucket=%s error=%s", bucket, e)
            raise

    def put_sync(self, key: str, data: bytes, content_type: str) -> str:
        buf = io.BytesIO(data)
        self.client.put_object(
            self.bucket, key, buf, length=len(data), content_type=content_type
        )
        url = self.client.presigned_get_object(self.bucket, key, expires=self.PRESIGNED_EXPIRY)
        return self._rewrite_url_if_needed(url)

    def get_sync(self, key: str) -> bytes:
        response = self.client.get_object(self.bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def delete_sync(self, key: str) -> None:
        self.client.remove_object(self.bucket, key)

    def _rewrite_url_if_needed(self, url: str) -> str:
        """Kalau public_endpoint berbeda dari internal endpoint (mis. MinIO di
        docker network 'minio:9000' tapi external access via 'storage.unhas.ac.id'),
        rewrite hostname di URL.
        """
        if self.public_endpoint == self._internal_endpoint():
            return url
        scheme = "https" if self.secure else "http"
        # URL dari minio client format: http://endpoint/bucket/key?X-Amz-...
        # Replace host bagian saja.
        try:
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(url)
            new = parsed._replace(netloc=self.public_endpoint, scheme=scheme)
            return urlunparse(new)
        except Exception:
            return url

    def _internal_endpoint(self) -> str:
        return getattr(self.client, "_base_url", None) or ""


_storage_instance: Storage | None = None


def get_storage() -> Storage:
    """Singleton storage based on STORAGE_BACKEND env."""
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance

    backend = os.getenv("STORAGE_BACKEND", "filesystem")
    if backend == "filesystem":
        path = os.getenv("STORAGE_LOCAL_PATH", "./storage")
        _storage_instance = FilesystemStorage(path)
        logger.info("storage_backend=filesystem path=%s", path)
    elif backend == "minio":
        endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "")
        secret_key = os.getenv("MINIO_SECRET_KEY", "")
        bucket = os.getenv("MINIO_BUCKET", "ragchat-images")
        secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        public_endpoint = os.getenv("MINIO_PUBLIC_ENDPOINT") or endpoint
        _storage_instance = MinIOStorage(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            secure=secure,
            public_endpoint=public_endpoint,
        )
        logger.info(
            "storage_backend=minio endpoint=%s public_endpoint=%s bucket=%s",
            endpoint, public_endpoint, bucket,
        )
    else:
        raise ValueError(f"Unknown STORAGE_BACKEND: {backend!r}")

    return _storage_instance
