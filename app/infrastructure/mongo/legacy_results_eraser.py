from bson import ObjectId

from app.infrastructure.mongo.client import Database


class LegacyResultsEraser:
    """Deletes the measurements stored by the previous version (``recognition_results``) when an account is erased."""

    def __init__(self, db: Database) -> None:
        self._results = db["recognition_results"]

    async def delete_all_for_user(self, user_id: str) -> int:
        if not ObjectId.is_valid(user_id):
            return 0
        result = await self._results.delete_many({"user_id": ObjectId(user_id)})
        return result.deleted_count
