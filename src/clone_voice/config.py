"""Configuration management for Clone Voice."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="CLONE_VOICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path.cwd())
    samples_dir: Path = Field(default=Path("samples"))
    models_dir: Path = Field(default=Path("models"))
    outputs_dir: Path = Field(default=Path("outputs"))

    # Model settings
    model_name: str = Field(default="tts_models/multilingual/multi-dataset/xtts_v2")
    device: Literal["cuda", "cpu", "auto"] = Field(default="auto")
    compute_type: Literal["float32", "float16", "int8"] = Field(default="float16")

    # Audio settings
    sample_rate: int = Field(default=22050)
    language: str = Field(default="en")

    # Recording settings
    recording_duration: int = Field(default=10, description="Default recording duration in seconds")
    recording_channels: int = Field(default=1)

    # Generation settings
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    length_penalty: float = Field(default=1.0, ge=0.5, le=2.0)
    repetition_penalty: float = Field(default=2.0, ge=1.0, le=10.0)
    top_k: int = Field(default=50, ge=1)
    top_p: float = Field(default=0.85, ge=0.0, le=1.0)

    def get_samples_path(self) -> Path:
        """Get absolute path to samples directory."""
        if self.samples_dir.is_absolute():
            return self.samples_dir
        return self.base_dir / self.samples_dir

    def get_models_path(self) -> Path:
        """Get absolute path to models directory."""
        if self.models_dir.is_absolute():
            return self.models_dir
        return self.base_dir / self.models_dir

    def get_outputs_path(self) -> Path:
        """Get absolute path to outputs directory."""
        if self.outputs_dir.is_absolute():
            return self.outputs_dir
        return self.base_dir / self.outputs_dir

    def get_device(self) -> str:
        """Get the compute device, auto-detecting if needed."""
        if self.device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings(**overrides) -> Settings:
    """Reload settings with optional overrides."""
    global settings
    settings = Settings(**overrides)
    return settings
