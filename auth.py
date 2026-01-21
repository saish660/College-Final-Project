from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from typing import Union

SECRET_KEY = "SMARTCLASSROOM"
ALGORITHM = "HS256"

pwd_context = CryptContext(
    # pbkdf2_sha256 has no 72-byte limit and no external C backend requirement
    schemes=["pbkdf2_sha256", "bcrypt_sha256", "bcrypt"],
    default="pbkdf2_sha256",
    deprecated="auto"
)

def _normalize_password(password: Union[str, bytes, bytearray]) -> str:
    if password is None:
        raise ValueError("Password is required")

    if isinstance(password, str):
        return password
    if isinstance(password, (bytes, bytearray)):
        return password.decode("utf-8", errors="ignore")

    raise TypeError("Password must be str or bytes")

def hash_password(password: str) -> str:
    return pwd_context.hash(_normalize_password(password))

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(_normalize_password(password), hashed)

def create_token(data: dict):
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + timedelta(hours=2)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
