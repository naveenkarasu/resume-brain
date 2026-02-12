import os
from pydantic_settings import BaseSettings


def _parse_cors_origins() -> list[str] | None:
    """Parse CORS_ORIGINS env var as comma-separated string or JSON list."""
    raw = os.environ.get("CORS_ORIGINS")
    if not raw:
        return None
    if raw.startswith("["):
        import json
        return json.loads(raw)
    return [o.strip() for o in raw.split(",") if o.strip()]


class Settings(BaseSettings):
    gemini_api_key: str = ""
    max_upload_size_mb: int = 5
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://tauri.localhost",
        "https://tauri.localhost",
        "tauri://localhost",
        "https://naveenkarasu.github.io",
    ]
    debug: bool = False

    # 6-model pipeline settings
    pipeline_mode: str = "legacy"  # "legacy" | "v2"
    pipeline_v2_strict: bool = False  # if True, don't fallback to legacy on v2 failure
    pipeline_model_dir: str = "training/models"  # directory for trained model artifacts

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "protected_namespaces": ("settings_",)}


_cors_override = _parse_cors_origins()
settings = Settings(**{"cors_origins": _cors_override} if _cors_override else {})
