"""
security.py  —  credentials for the RentWise API.

* Passwords: scrypt (stdlib hashlib, memory-hard) with a random 16-byte salt per user;
  verified in constant time. Stored as  scrypt$n$r$p$salt_hex$hash_hex.
* API keys and session tokens: 256-bit random secrets (secrets.token_urlsafe). Only a
  SHA-256 hash is stored; the plaintext is shown to the user exactly once. A high-
  entropy random secret does not need a slow hash — SHA-256 of it is not guessable.
* API keys carry a short public prefix ("rw_" + 8 chars) so a user can tell keys apart
  in listings without the secret ever being stored or shown again.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1
MIN_PASSWORD_LEN = 10
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt_hex, hash_hex = stored.split("$")
        if algo != "scrypt":
            return False
        h = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex),
                           n=int(n), r=int(r), p=int(p), dklen=len(hash_hex) // 2)
        return hmac.compare_digest(h.hex(), hash_hex)
    except (ValueError, TypeError):
        return False


def validate_email(email: str) -> str:
    email = (email or "").strip().lower()
    if len(email) > 254 or not EMAIL_RE.match(email):
        raise ValueError("invalid email address")
    return email


def validate_password(password: str) -> None:
    if not password or len(password) < MIN_PASSWORD_LEN:
        raise ValueError(f"password must be at least {MIN_PASSWORD_LEN} characters")
    if len(password) > 256:
        raise ValueError("password too long")


def sha256(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def new_api_key() -> tuple[str, str, str]:
    """(plaintext_key, public_prefix, sha256_hash)."""
    body = secrets.token_urlsafe(32)
    key = f"rw_{body}"
    return key, key[:11], sha256(key)


def new_session_token() -> tuple[str, str]:
    """(plaintext_token, sha256_hash)."""
    tok = secrets.token_urlsafe(32)
    return tok, sha256(tok)
