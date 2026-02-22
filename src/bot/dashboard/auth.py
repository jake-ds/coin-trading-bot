"""JWT authentication for the dashboard API.

Auth is disabled by default when dashboard_password='changeme' (dev mode).
Enable by setting a real password in DASHBOARD_PASSWORD env var.
"""

import secrets
import time

import structlog
from jose import JWTError, jwt

logger = structlog.get_logger()

# Token expiry (seconds)
ACCESS_TOKEN_EXPIRE = 3600  # 1 hour
REFRESH_TOKEN_EXPIRE = 7 * 24 * 3600  # 7 days

ALGORITHM = "HS256"

# In-memory blacklist for invalidated refresh tokens
_refresh_blacklist: set[str] = set()


def _get_jwt_secret(settings) -> str:
    """Get JWT secret from settings or generate one."""
    if hasattr(settings, "jwt_secret") and settings.jwt_secret:
        return settings.jwt_secret
    return "dev-secret-not-for-production"


def is_auth_enabled(settings) -> bool:
    """Check if authentication is enabled (password changed from default)."""
    password = getattr(settings, "dashboard_password", "changeme")
    return password != "changeme"


def verify_credentials(settings, username: str, password: str) -> bool:
    """Verify username and password against configured credentials."""
    expected_user = getattr(settings, "dashboard_username", "admin")
    expected_pass = getattr(settings, "dashboard_password", "changeme")
    return username == expected_user and password == expected_pass


def create_access_token(settings, username: str) -> str:
    """Create a JWT access token."""
    secret = _get_jwt_secret(settings)
    payload = {
        "sub": username,
        "type": "access",
        "exp": int(time.time()) + ACCESS_TOKEN_EXPIRE,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def create_refresh_token(settings, username: str) -> str:
    """Create a JWT refresh token with a unique jti for blacklisting."""
    secret = _get_jwt_secret(settings)
    jti = secrets.token_hex(16)
    payload = {
        "sub": username,
        "type": "refresh",
        "jti": jti,
        "exp": int(time.time()) + REFRESH_TOKEN_EXPIRE,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_token(settings, token: str) -> dict | None:
    """Decode and validate a JWT token. Returns claims or None if invalid."""
    secret = _get_jwt_secret(settings)
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
        # Check expiry
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except JWTError:
        return None


def validate_access_token(settings, token: str) -> str | None:
    """Validate an access token and return the username, or None if invalid."""
    claims = decode_token(settings, token)
    if claims is None:
        return None
    if claims.get("type") != "access":
        return None
    return claims.get("sub")


def validate_refresh_token(settings, token: str) -> str | None:
    """Validate a refresh token and return the username, or None if invalid/blacklisted."""
    claims = decode_token(settings, token)
    if claims is None:
        return None
    if claims.get("type") != "refresh":
        return None
    # Check blacklist
    jti = claims.get("jti")
    if jti and jti in _refresh_blacklist:
        return None
    return claims.get("sub")


def blacklist_refresh_token(settings, token: str) -> bool:
    """Blacklist a refresh token (logout). Returns True if successful."""
    claims = decode_token(settings, token)
    if claims is None:
        return False
    jti = claims.get("jti")
    if jti:
        _refresh_blacklist.add(jti)
        return True
    return False


def clear_blacklist() -> None:
    """Clear the refresh token blacklist (for testing)."""
    _refresh_blacklist.clear()
