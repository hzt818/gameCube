"""
Configuration module for AI Agent Core System.
Loads settings from environment variables using Pydantic Settings.
"""
import logging
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    All settings can be overridden via .env file or environment variables.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    app_name: str = Field(default="AI Agent Core", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode flag")
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Running environment"
    )
    
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_workers: int = Field(default=1, description="Number of API workers")
    
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/agent_core",
        description="PostgreSQL database URL"
    )
    database_pool_size: int = Field(default=10, description="Database connection pool size")
    database_max_overflow: int = Field(default=20, description="Database max overflow connections")
    
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    redis_password: str | None = Field(default=None, description="Redis password")
    
    qdrant_host: str = Field(default="localhost", description="Qdrant server host")
    qdrant_port: int = Field(default=6333, description="Qdrant server port")
    qdrant_collection: str = Field(default="agent_memory", description="Qdrant collection name")
    qdrant_api_key: str | None = Field(default=None, description="Qdrant API key")
    
    vllm_base_url: str = Field(
        default="http://localhost:8001/v1",
        description="vLLM API base URL"
    )
    vllm_api_key: str | None = Field(default=None, description="vLLM API key")
    vllm_model: str = Field(default="default", description="vLLM model name")
    vllm_max_tokens: int = Field(default=4096, description="Maximum tokens for generation")
    vllm_temperature: float = Field(default=0.7, description="Generation temperature")
    vllm_timeout: float = Field(default=60.0, description="vLLM request timeout in seconds")
    
    agent_max_iterations: int = Field(default=10, description="Maximum agent iterations")
    agent_max_tool_calls: int = Field(default=50, description="Maximum tool calls per session")
    agent_tool_timeout: float = Field(default=30.0, description="Tool execution timeout in seconds")
    agent_reflection_enabled: bool = Field(default=True, description="Enable self-reflection")
    
    memory_short_term_max_messages: int = Field(default=100, description="Max messages in short-term memory")
    memory_compression_threshold: int = Field(default=50, description="Threshold for memory compression")
    memory_vector_dimension: int = Field(default=1536, description="Vector embedding dimension")
    
    rag_top_k: int = Field(default=5, description="Top K results for RAG retrieval")
    rag_similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold")
    
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of allowed values."""
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are only loaded once.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()


def setup_logging(settings: Settings | None = None) -> logging.Logger:
    """
    Configure application logging based on settings.
    
    Args:
        settings: Optional settings instance. If None, loads from get_settings()
        
    Returns:
        logging.Logger: Configured root logger
    """
    if settings is None:
        settings = get_settings()
    
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=settings.log_format,
        force=True
    )
    
    logger = logging.getLogger(settings.app_name)
    logger.setLevel(getattr(logging, settings.log_level))
    
    return logger


settings = get_settings()
logger = setup_logging(settings)
