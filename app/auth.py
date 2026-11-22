"""Login for the whole site: one shared password, then a signed cookie."""

import hashlib
import hmac
import secrets
import time


class SessionAuth:
    """Checks the site password and issues login tokens for the session cookie.

    A token is "<expiry time>.<signature>". The signature is an HMAC (a
    fingerprint only someone holding the key can produce) of the expiry time,
    so a visitor can't edit the expiry or invent a token without the key.
    Nothing is stored on the server: the token proves itself, so logins
    survive a restart.
    """

    def __init__(self, password: str, session_days: int) -> None:
        """Set up auth for one password.

        Args:
            password: The site password, already validated by Settings.
            session_days: How long a login lasts before asking again.
        """
        self._password = password.encode()
        self._session_seconds = session_days * 24 * 60 * 60
        # The signing key comes from the password, so changing APP_PASSWORD
        # logs everyone out. scrypt is deliberately slow (~50 ms): someone who
        # steals a cookie can't cheaply try millions of passwords against it.
        self._key = hashlib.scrypt(self._password, salt=b"rostee-session", n=2**14, r=8, p=1)

    def login(self, attempt: str) -> str | None:
        """Return a fresh token if the password is right, otherwise None."""
        # compare_digest takes the same time whether the first or the last
        # character is wrong, so response timing doesn't leak the password.
        if not secrets.compare_digest(attempt.encode(), self._password):
            return None
        expires = str(int(time.time()) + self._session_seconds)
        return f"{expires}.{self._sign(expires)}"

    def is_valid(self, token: str) -> bool:
        """True only for an unexpired token this server signed."""
        expires, _, signature = token.partition(".")
        if not expires.isdigit():
            return False
        if not secrets.compare_digest(signature, self._sign(expires)):
            return False
        return int(expires) > time.time()

    def _sign(self, message: str) -> str:
        return hmac.new(self._key, message.encode(), hashlib.sha256).hexdigest()
