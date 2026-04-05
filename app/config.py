"""
app/config.py — 集中式設定管理
================================
使用 pydantic-settings 從 .env 自動讀取並驗證所有環境變數。
所有模組皆透過 `get_settings()` 取得單例設定物件，避免重複解析。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project Ē-Kóng 全域設定。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LINE ──────────────────────────────────────────────────────────────────
    line_channel_access_token: str = Field(..., description="LINE Channel Access Token")
    line_channel_secret: str = Field(..., description="LINE Channel Secret")

    # ── ngrok ─────────────────────────────────────────────────────────────────
    ngrok_auth_token: str = Field(..., description="ngrok Auth Token")

    # ── FastAPI Server ────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── LLM ───────────────────────────────────────────────────────────
    llm_provider: Literal["local", "gemini", "hybrid"] = Field(
        default="local",
        description="LLM 提供者：local, gemini, 或 hybrid (本地超時切 API)",
    )
    google_api_key: str | None = Field(default=None, description="Google AI Studio API Key")

    llm_model_path: str = Field(
        default="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        description="GGUF 模型絕對路徑 (建議 1.5B 以利 CPU 運作)",
    )
    llm_local_timeout: int = Field(default=60, description="本地推論超時時間（秒），僅 hybrid 模式有效")
    llm_n_gpu_layers: int = Field(default=35, description="GPU offload 層數；-1 表示全量")
    llm_max_tokens: int = 512
    llm_temperature: float = 0.7

    # ── Whisper STT ───────────────────────────────────────────────────────────
    whisper_model_size: Literal["tiny", "base", "small", "medium", "large-v3"] = "base"
    whisper_device: Literal["cpu", "cuda"] = "cpu"
    whisper_compute_type: Literal["int8", "float16", "float32"] = "int8"

    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature 必須在 [0.0, 2.0] 範圍內")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """回傳全域設定單例（使用 lru_cache 確保只解析一次）。"""
    return Settings()  # type: ignore[call-arg]
