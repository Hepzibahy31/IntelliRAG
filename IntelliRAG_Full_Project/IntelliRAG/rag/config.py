"""Settings: environment variables and an optional .env file."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_env_file():
    path = BASE_DIR / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if value:                       # empty values never override
                os.environ.setdefault(key.strip(), value)


_load_env_file()


def get(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


DATA_DIR = Path(get("DATA_DIR", str(BASE_DIR / "data")))
STORAGE_DIR = Path(get("STORAGE_DIR", str(BASE_DIR / "storage")))
MAX_UPLOAD_MB = int(get("MAX_UPLOAD_MB", "30"))
