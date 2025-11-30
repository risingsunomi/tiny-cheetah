from __future__ import annotations

import secrets


def generate_identity(username: str) -> dict:
    """Return a simple identity payload with a fingerprint."""
    return {
        "username": username,
        "fingerprint": secrets.token_hex(8),
        "pgp_public": "",
        "pgp_private": "",
    }
