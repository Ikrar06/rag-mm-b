"""Auth router — login, logout, me endpoints."""

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from backend.services.auth import authenticate_user, create_token, decode_token

router = APIRouter(prefix="/api/auth", tags=["auth"])

_COOKIE_NAME = "access_token"
_COOKIE_MAX_AGE = 24 * 3600  # 24 jam


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str
    name: str
    role: str
    nim: str | None = None


class UserInfo(BaseModel):
    username: str
    name: str
    role: str
    nim: str | None = None


def _get_token_from_request(request: Request) -> str | None:
    token = request.cookies.get(_COOKIE_NAME)
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
    return token


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, response: Response):
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


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(_COOKIE_NAME)
    return {"status": "ok"}


@router.get("/me", response_model=UserInfo)
async def me(request: Request):
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
