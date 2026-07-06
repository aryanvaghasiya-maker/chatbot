import hashlib
import json
from pathlib import Path

# Resolve cache directory relative to the workspace root
CACHE_DIR = Path(__file__).resolve().parents[3] / "resume_cache"
CACHE_DIR.mkdir(exist_ok=True)

class ResumeCache:
    CACHE_DIR = CACHE_DIR

    @staticmethod
    def key(file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    @staticmethod
    def exists(key: str) -> bool:
        return (CACHE_DIR / f"{key}.json").exists()

    @staticmethod
    def load(key: str) -> dict:
        path = CACHE_DIR / f"{key}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def save(key: str, data: dict) -> None:
        path = CACHE_DIR / f"{key}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
