from pwdlib import PasswordHash
from pwdlib.exceptions import UnknownHashError
from pwdlib.hashers.argon2 import Argon2Hasher
from pwdlib.hashers.bcrypt import BcryptHasher

from app.domain.auth import PasswordCheck


class PwdlibPasswordHasher:
    """Argon2id for new hashes; legacy bcrypt hashes still verify and are flagged for re-hashing."""

    def __init__(self) -> None:
        self._hasher = PasswordHash((Argon2Hasher(), BcryptHasher()))
        self._decoy = self._hasher.hash("tensoscan-decoy-password")

    def hash(self, password: str) -> str:
        return self._hasher.hash(password)

    def check(self, password: str, password_hash: str) -> PasswordCheck:
        try:
            valid, updated_hash = self._hasher.verify_and_update(password, password_hash)
        except UnknownHashError:
            return PasswordCheck(valid=False)
        return PasswordCheck(valid=valid, needs_rehash=valid and updated_hash is not None)

    def burn(self, password: str) -> None:
        self._hasher.verify(password, self._decoy)
