"""Configuration for ML Service."""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """ML Service settings."""

    # Service configuration
    service_name: str = "ml-service"
    debug: bool = False

    # Model configuration
    deepgaze_model: str = "deepgaze3"  # deepgaze2, deepgaze2e, deepgaze3
    device: str = "cpu"  # cpu or cuda
    model_cache_dir: str = "./models"

    # Image processing
    max_image_size: int = 1024  # Max dimension for input images
    output_format: str = "png"

    # Performance
    request_timeout: int = 60
    max_batch_size: int = 4

    class Config:
        env_prefix = "ML_"
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
