import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the ingester/ project root (one level above src/)
_env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(_env_path)


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and fill it in."
        )
    return url
