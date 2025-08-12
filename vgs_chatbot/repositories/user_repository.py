"""User repository for database operations."""


from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vgs_chatbot.models.user import User

from .database import UserTable


class UserRepository:
    """Repository for user database operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize user repository.

        Args:
            session: Database session
        """
        self.session = session

    async def get_by_username(self, username: str) -> User | None:
        """Get user by username.

        Args:
            username: Username to search for

        Returns:
            User if found, None otherwise
        """
        stmt = select(UserTable).where(UserTable.username == username)
        result = await self.session.execute(stmt)
        user_row = result.scalar_one_or_none()

        if user_row:
            return User.model_validate(user_row)
        return None

    async def get_by_id(self, user_id: int) -> User | None:
        """Get user by ID.

        Args:
            user_id: User ID to search for

        Returns:
            User if found, None otherwise
        """
        stmt = select(UserTable).where(UserTable.id == user_id)
        result = await self.session.execute(stmt)
        user_row = result.scalar_one_or_none()

        if user_row:
            return User.model_validate(user_row)
        return None

    async def create(self, user: User) -> User:
        """Create new user.

        Args:
            user: User to create

        Returns:
            Created user with ID
        """
        user_table = UserTable(
            username=user.username,
            email=user.email,
            password_hash=user.password_hash,
            is_active=user.is_active
        )

        self.session.add(user_table)
        await self.session.commit()
        await self.session.refresh(user_table)

        return User.model_validate(user_table)

    async def update_last_login(self, user_id: int) -> None:
        """Update user's last login timestamp.

        Args:
            user_id: User ID to update
        """
        stmt = select(UserTable).where(UserTable.id == user_id)
        result = await self.session.execute(stmt)
        user_row = result.scalar_one_or_none()

        if user_row:
            from datetime import datetime
            user_row.last_login = datetime.utcnow()
            await self.session.commit()
