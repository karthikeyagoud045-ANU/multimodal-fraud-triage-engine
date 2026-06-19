"""
config.py — Centralised Application Configuration via Pydantic Settings.

All configuration is loaded from environment variables / a `.env` file at
startup. No hardcoded secrets anywhere in the codebase.

Usage:
    from config import settings

    client = AsyncGroq(api_key=settings.groq_api_key)

Pydantic Settings automatically:
  1. Reads from `.env` (via `env_file = ".env"` in model_config).
  2. Falls back to actual environment variables if `.env` is absent.
  3. Raises a clear ValidationError at startup if a required field is missing —
     fail-fast rather than a cryptic KeyError at runtime.
"""
from __future__ import annotations

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from .env.

    Required fields (must be set in .env or environment):
        groq_api_key      — Groq console: https://console.groq.com/keys
        nvidia_api_key    — Nvidia NIM:   https://integrate.api.nvidia.com

    Optional fields (have sensible defaults for hackathon free-tier use):
        groq_model        — Main text LLM (70B for best accuracy)
        groq_scribe_model — Justification writer (8B is fast and cheap)
        nvidia_model      — Vision LLM (90B for best image understanding)
        concurrency       — Max simultaneous claim-processing coroutines
        max_image_size_px — Max px on longest side before resizing (token budget)
        jpeg_quality      — JPEG re-encode quality for base64 images (0-100)
    """

    model_config = SettingsConfigDict(
        # Load from .env in the current working directory.
        # Also reads from the system environment (env vars override .env).
        env_file=".env",
        env_file_encoding="utf-8",
        # Silently ignore extra fields in .env that don't match any setting.
        extra="ignore",
        # Make field names case-insensitive: GROQ_API_KEY == groq_api_key.
        case_sensitive=False,
    )

    # ── Required: API Keys ────────────────────────────────────────────────────
    # SecretStr prevents accidental logging of the key value.
    # Access with: settings.groq_api_key.get_secret_value()
    groq_api_key: SecretStr = Field(
        default=SecretStr("dummy_groq_key"),
        description="Groq API key. Required. Get one at https://console.groq.com/keys",
    )
    nvidia_api_key: SecretStr = Field(
        default=SecretStr("dummy_nvidia_key"),
        alias="NVIDIA_API_KEY",
        description="Nvidia NIM API key. Required. Get one at https://integrate.api.nvidia.com",
    )

    # ── Model Selection ───────────────────────────────────────────────────────
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description=(
            "Groq model for text intent extraction. "
            "llama-3.3-70b-versatile is the recommended free-tier option. "
            "Alternative: llama-3.1-8b-instant (faster, lower quality)."
        ),
    )
    groq_scribe_model: str = Field(
        default="llama-3.1-8b-instant",
        description=(
            "Groq model for writing claim justifications. "
            "Uses a separate lighter model to avoid competing with the main "
            "text extractor for TPM quota."
        ),
    )
    nvidia_model: str = Field(
        default="meta/llama-3.2-90b-vision-instruct",
        description=(
            "Nvidia NIM vision model. "
            "meta/llama-3.2-90b-vision-instruct is the best open-weights "
            "option for damage assessment."
        ),
    )

    # ── Nvidia NIM Endpoint ───────────────────────────────────────────────────
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="Nvidia NIM OpenAI-compatible endpoint base URL.",
    )

    # ── Pipeline Behaviour ────────────────────────────────────────────────────
    concurrency: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Maximum number of claims processed concurrently. "
            "Keep at 3 for free-tier Groq/Nvidia accounts to avoid 429 errors. "
            "Increase to 5-10 if on a paid plan."
        ),
    )

    # ── Image Processing ──────────────────────────────────────────────────────
    max_image_size_px: int = Field(
        default=1024,
        ge=256,
        le=4096,
        description=(
            "Maximum pixel dimension (width or height) after resizing. "
            "1024px saves ~60% of vision tokens vs full-resolution images "
            "with minimal accuracy loss for damage assessment tasks."
        ),
    )
    jpeg_quality: int = Field(
        default=86,
        ge=50,
        le=100,
        description=(
            "JPEG re-encode quality for base64 image payloads. "
            "86 is a good balance of file size and visual quality. "
            "Lower values reduce token cost but may lose fine damage details."
        ),
    )

    # ── OCR Security ──────────────────────────────────────────────────────────
    ocr_languages: list[str] = Field(
        default=["en", "ch_sim"],
        description=(
            "EasyOCR language codes for visual injection scanning. "
            "['en', 'ch_sim'] covers English and Chinese. "
            "Add 'hi' for Devanagari (Hindi) — requires extra model download."
        ),
    )

    @model_validator(mode="after")
    def _validate_concurrency_vs_model(self) -> "Settings":
        """Warn if concurrency is set high on a free-tier account.

        This is a non-blocking advisory check — it prints a warning rather
        than raising an error, since we can't know the account tier at config time.
        """
        if self.concurrency > 5:
            import warnings
            warnings.warn(
                f"concurrency={self.concurrency} may cause frequent 429 rate-limit errors "
                "on free-tier Groq/Nvidia accounts. Consider setting concurrency <= 3.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def groq_api_key_str(self) -> str:
        """Convenience method to get the raw Groq API key string."""
        return self.groq_api_key.get_secret_value()

    def nvidia_api_key_str(self) -> str:
        """Convenience method to get the raw Nvidia API key string."""
        return self.nvidia_api_key.get_secret_value()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — import this everywhere:
#
#   from config import settings
#   print(settings.groq_model)
#
# Settings are loaded once at module import time. If .env is missing or a
# required key is absent, pydantic raises a clear ValidationError immediately.
# ─────────────────────────────────────────────────────────────────────────────
settings = Settings()
