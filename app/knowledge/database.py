"""
Database module for AI Agent Core System.
Manages database connections and sessions.
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.config import settings

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Exception raised for database errors."""
    pass


class Database:
    """
    Database connection manager.
    
    Features:
    - Async connection pooling
    - Session management
    - Health checks
    - Connection lifecycle
    """
    
    def __init__(
        self,
        database_url: str | None = None,
        pool_size: int | None = None,
        max_overflow: int | None = None,
        echo: bool = False
    ) -> None:
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection URL
            pool_size: Connection pool size
            max_overflow: Max overflow connections
            echo: Whether to echo SQL statements
        """
        self.database_url = database_url or settings.database_url
        self.pool_size = pool_size or settings.database_pool_size
        self.max_overflow = max_overflow or settings.database_max_overflow
        self.echo = echo or settings.debug
        
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker | None = None
        self._initialized = False
        
        logger.info(
            f"Database configured: pool_size={self.pool_size}, "
            f"max_overflow={self.max_overflow}"
        )
    
    async def initialize(self) -> None:
        """Initialize database connection pool."""
        if self._initialized:
            return
        
        try:
            self._engine = create_async_engine(
                self.database_url,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                echo=self.echo,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False
            )
            
            async with self._engine.connect() as conn:
                await conn.execute("SELECT 1")
            
            self._initialized = True
            logger.info("Database connection initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    async def close(self) -> None:
        """Close database connection pool."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._initialized = False
            logger.info("Database connection closed")
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session context manager.
        
        Yields:
            AsyncSession: Database session
            
        Example:
            async with db.session() as session:
                result = await session.execute(query)
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._session_factory:
            raise DatabaseError("Database not initialized")
        
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()
    
    async def get_session(self) -> AsyncSession:
        """
        Get a database session.
        
        Note: Caller is responsible for committing/rollback and closing.
        
        Returns:
            AsyncSession: Database session
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._session_factory:
            raise DatabaseError("Database not initialized")
        
        return self._session_factory()
    
    @property
    def engine(self) -> AsyncEngine:
        """Get database engine."""
        if not self._engine:
            raise DatabaseError("Database not initialized")
        return self._engine
    
    async def health_check(self) -> dict[str, any]:
        """
        Check database health.
        
        Returns:
            Dictionary with health status
        """
        if not self._initialized or not self._engine:
            return {
                "status": "not_initialized",
                "healthy": False
            }
        
        try:
            async with self._engine.connect() as conn:
                await conn.execute("SELECT 1")
            
            return {
                "status": "healthy",
                "healthy": True,
                "pool_size": self.pool_size
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e)
            }
    
    async def execute_raw(self, sql: str, params: dict | None = None) -> any:
        """
        Execute raw SQL query.
        
        Args:
            sql: SQL query
            params: Query parameters
            
        Returns:
            Query result
        """
        if not self._initialized:
            await self.initialize()
        
        async with self.session() as session:
            from sqlalchemy import text
            result = await session.execute(text(sql), params or {})
            return result.fetchall()
    
    def get_stats(self) -> dict[str, any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with stats
        """
        return {
            "initialized": self._initialized,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "database_url": self.database_url.split("@")[-1] if "@" in self.database_url else "hidden"
        }
    
    def __repr__(self) -> str:
        return f"Database(initialized={self._initialized})"


_database: Database | None = None


async def get_database() -> Database:
    """
    Get the global database instance.
    
    Returns:
        Database instance
    """
    global _database
    if _database is None:
        _database = Database()
        await _database.initialize()
    return _database


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database session.
    
    Yields:
        AsyncSession: Database session
    """
    db = await get_database()
    async with db.session() as session:
        yield session
