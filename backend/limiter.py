"""Rate limiter singleton — di-share antara main.py dan routers."""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
