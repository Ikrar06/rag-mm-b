"""Storage abstraction — POC stub.

Saat ini: filesystem storage (simpan di ./storage/).
Uncomment MinIOStorage saat deploy ke POC dengan MinIO.

Requires (POC):
  pip install minio aiofiles

Swap cukup dengan ubah env var: STORAGE_BACKEND=minio
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path


class Storage(ABC):
    @abstractmethod
    async def put(self, key: str, data: bytes, content_type: str) -> str: ...

    @abstractmethod
    async def get(self, key: str) -> bytes: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...


# =============================================================================
# Dev: filesystem storage
# =============================================================================

class FilesystemStorage(Storage):
    def __init__(self, base: str):
        self.base = Path(base)
        self.base.mkdir(parents=True, exist_ok=True)

    async def put(self, key: str, data: bytes, content_type: str) -> str:
        import aiofiles  # pip install aiofiles
        path = self.base / key
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "wb") as f:
            await f.write(data)
        return f"/api/files/{key}"

    async def get(self, key: str) -> bytes:
        import aiofiles
        async with aiofiles.open(self.base / key, "rb") as f:
            return await f.read()

    async def delete(self, key: str) -> None:
        path = self.base / key
        if path.exists():
            path.unlink()


# =============================================================================
# POC: MinIO storage — Uncomment saat STORAGE_BACKEND=minio
# =============================================================================

# class MinIOStorage(Storage):
#     def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
#         from minio import Minio
#         self.client = Minio(endpoint, access_key, secret_key, secure=False)
#         self.bucket = bucket
#         if not self.client.bucket_exists(bucket):
#             self.client.make_bucket(bucket)
#
#     async def put(self, key: str, data: bytes, content_type: str) -> str:
#         import io
#         from minio.commonconfig import CopySource
#         # minio client is sync — run in executor for async context
#         import asyncio
#         loop = asyncio.get_event_loop()
#         buf = io.BytesIO(data)
#         await loop.run_in_executor(
#             None,
#             lambda: self.client.put_object(self.bucket, key, buf, len(data), content_type=content_type)
#         )
#         return f"minio://{self.bucket}/{key}"
#
#     async def get(self, key: str) -> bytes:
#         import asyncio
#         loop = asyncio.get_event_loop()
#         response = await loop.run_in_executor(
#             None, lambda: self.client.get_object(self.bucket, key)
#         )
#         return response.read()
#
#     async def delete(self, key: str) -> None:
#         import asyncio
#         loop = asyncio.get_event_loop()
#         await loop.run_in_executor(None, lambda: self.client.remove_object(self.bucket, key))


def get_storage() -> Storage:
    backend = os.getenv("STORAGE_BACKEND", "filesystem")
    if backend == "filesystem":
        path = os.getenv("STORAGE_LOCAL_PATH", "./storage")
        return FilesystemStorage(path)
    # elif backend == "minio":
    #     from backend.config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET
    #     return MinIOStorage(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET)
    raise ValueError(f"Unknown STORAGE_BACKEND: {backend!r}")
