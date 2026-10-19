import logging

from pymongo.errors import DuplicateKeyError, OperationFailure

from app.infrastructure.mongo.client import Database

logger = logging.getLogger(__name__)


async def normalize_legacy_emails(db: Database) -> None:
    """Lower-cases emails written by the previous version (which stored them as typed)."""
    await db["users"].update_many({}, [{"$set": {"email": {"$toLower": {"$trim": {"input": "$email"}}}}}])


async def ensure_indexes(db: Database) -> None:
    """Start-up migration: normalise legacy data, then create every index."""
    try:
        await normalize_legacy_emails(db)
    except OperationFailure:
        logger.exception("Could not normalise legacy emails")
    await create_indexes(db)


async def create_indexes(db: Database) -> None:
    try:
        await db["users"].create_index("email", unique=True, name="users_email_unique")
    except (DuplicateKeyError, OperationFailure):
        # Two legacy accounts that differ only by case. The service keeps running; the duplicate has to be
        # resolved by hand before uniqueness can be enforced by the database.
        logger.exception("Could not enforce unique emails; resolve the duplicated accounts")

    refresh = db["refresh_tokens"]
    await refresh.create_index("token_hash", unique=True, name="refresh_token_hash_unique")
    await refresh.create_index("family_id", name="refresh_family")
    await refresh.create_index("user_id", name="refresh_user")
    await refresh.create_index("expires_at", expireAfterSeconds=0, name="refresh_expiry_ttl")
