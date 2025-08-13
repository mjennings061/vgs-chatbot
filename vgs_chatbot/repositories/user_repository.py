"""User repository for database operations."""

from datetime import UTC, datetime

from bson import ObjectId
from pymongo.collection import Collection

from vgs_chatbot.models.user import User


class UserRepository:
    """Repository for user database operations."""

    def __init__(self, collection: Collection) -> None:
        """Initialize user repository.

        Args:
            collection: MongoDB users collection
        """
        self.collection = collection

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email.

        Args:
            email: Email to search for

        Returns:
            User if found, None otherwise
        """
        user_doc = self.collection.find_one({"email": email})

        if user_doc:
            user_doc["_id"] = str(user_doc["_id"])
            return User.model_validate(user_doc)
        return None

    async def get_by_id(self, user_id: str) -> User | None:
        """Get user by ID.

        Args:
            user_id: User ID to search for

        Returns:
            User if found, None otherwise
        """
        try:
            object_id = ObjectId(user_id)
            user_doc = self.collection.find_one({"_id": object_id})

            if user_doc:
                user_doc["_id"] = str(user_doc["_id"])
                return User.model_validate(user_doc)
        except Exception:
            return None
        return None

    async def create(self, user: User) -> User:
        """Create new user.

        Args:
            user: User to create

        Returns:
            Created user with ID
        """
        user_dict = user.model_dump(exclude={"id"}, by_alias=True)

        result = self.collection.insert_one(user_dict)
        user_dict["_id"] = str(result.inserted_id)

        return User.model_validate(user_dict)

    async def update_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp.

        Args:
            user_id: User ID to update
        """
        try:
            object_id = ObjectId(user_id)
            self.collection.update_one(
                {"_id": object_id},
                {"$set": {"last_login": datetime.now(UTC)}}
            )
        except Exception:
            pass
