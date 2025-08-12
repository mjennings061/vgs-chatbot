"""Database models and configuration."""

from collections.abc import AsyncGenerator
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for database models."""
    pass


class UserTable(Base):
    """User table model."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class DatabaseManager:
    """Database connection manager."""

    def __init__(self, database_url: str) -> None:
        """Initialize database manager.

        Args:
            database_url: Database connection URL
        """
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def create_tables(self) -> None:
        """Create database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session.

        Yields:
            Database session
        """
        async with self.async_session() as session:
            try:
                yield session
            finally:
                await session.close()

    async def close(self) -> None:
        """Close database connection."""
        await self.engine.dispose()
