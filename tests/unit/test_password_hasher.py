import bcrypt

from app.infrastructure.security.password_hasher import PwdlibPasswordHasher

sut = PwdlibPasswordHasher()


def test_new_hashes_use_argon2id_and_verify() -> None:
    password_hash = sut.hash("s3cret-pass")

    assert password_hash.startswith("$argon2id$")
    assert sut.check("s3cret-pass", password_hash).valid
    assert not sut.check("wrong", password_hash).valid


def test_a_current_hash_does_not_need_rehashing() -> None:
    assert not sut.check("s3cret-pass", sut.hash("s3cret-pass")).needs_rehash


def test_a_legacy_bcrypt_hash_verifies_and_asks_to_be_rehashed() -> None:
    legacy = bcrypt.hashpw(b"s3cret-pass", bcrypt.gensalt(rounds=4)).decode()

    check = sut.check("s3cret-pass", legacy)

    assert check.valid
    assert check.needs_rehash


def test_a_wrong_password_against_a_legacy_hash_is_rejected_without_rehash() -> None:
    legacy = bcrypt.hashpw(b"s3cret-pass", bcrypt.gensalt(rounds=4)).decode()

    check = sut.check("wrong", legacy)

    assert not check.valid
    assert not check.needs_rehash


def test_an_unknown_hash_format_is_just_invalid() -> None:
    assert not sut.check("anything", "plaintext-garbage").valid
    assert not sut.check("anything", "").valid
