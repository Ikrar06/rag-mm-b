"""Auth router — login, logout, me endpoints."""

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from backend.services.auth import authenticate_user, create_token, decode_token

router = APIRouter(prefix="/api/auth", tags=["auth"])

_COOKIE_NAME = "access_token"
_COOKIE_MAX_AGE = 24 * 3600  # 24 jam


class LoginRequest(BaseModel):
    username: str = Field(..., description="Username akun (lihat users.json untuk akun dev)")
    password: str = Field(..., description="Password akun")


class LoginResponse(BaseModel):
    token: str = Field(..., description="JWT Bearer token — bisa dipakai di header Authorization: Bearer <token>")
    username: str = Field(..., description="Username yang berhasil login")
    name: str = Field(..., description="Nama lengkap user")
    role: str = Field(..., description="Role: public | mahasiswa | staf_akademik | calon_mahasiswa")
    nim: str | None = Field(None, description="NIM mahasiswa (hanya untuk role mahasiswa)")


class UserInfo(BaseModel):
    username: str
    name: str
    role: str = Field(..., description="Role: public | mahasiswa | staf_akademik | calon_mahasiswa")
    nim: str | None = None


def _get_token_from_request(request: Request) -> str | None:
    token = request.cookies.get(_COOKIE_NAME)
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
    return token


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Login dan dapatkan JWT token",
    responses={401: {"description": "Username atau password salah"}},
)
async def login(body: LoginRequest, response: Response):
    """
    Login dengan username + password.

    - JWT token dikembalikan di response body **dan** di-set sebagai httpOnly cookie (`access_token`).
    - Untuk integrasi BE: gunakan token dari response body di header `Authorization: Bearer <token>`.
    - Untuk frontend: cookie otomatis dikirim browser pada setiap request.
    - Token berlaku selama **24 jam**.
    """
    user = authenticate_user(body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Username atau password salah")

    token = create_token(user)

    # Set httpOnly cookie untuk keamanan XSS
    response.set_cookie(
        key=_COOKIE_NAME,
        value=token,
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )

    return LoginResponse(
        token=token,
        username=user["username"],
        name=user.get("name", user["username"]),
        role=user["role"],
        nim=user.get("nim"),
    )


@router.post("/logout", summary="Logout — hapus session cookie")
async def logout(response: Response):
    """Hapus cookie `access_token`. Tidak perlu auth header."""
    response.delete_cookie(_COOKIE_NAME)
    return {"status": "ok"}


@router.get(
    "/me",
    response_model=UserInfo,
    summary="Cek user yang sedang login",
    responses={401: {"description": "Belum login atau token expired"}},
)
async def me(request: Request):
    """
    Kembalikan info user dari JWT token.

    - Auth: cookie `access_token` **atau** header `Authorization: Bearer <token>`.
    - Dipakai frontend untuk cek apakah user masih login setelah refresh halaman.
    """
    token = _get_token_from_request(request)
    if not token:
        raise HTTPException(status_code=401, detail="Belum login")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Token tidak valid atau sudah expired")
    return UserInfo(
        username=payload["sub"],
        name=payload["name"],
        role=payload["role"],
        nim=payload.get("nim"),
    )
