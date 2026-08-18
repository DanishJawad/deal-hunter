from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppConfig:
    chroma_collection_name: str
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embed_model: str
    cheapshark_base_url: str
    cheapshark_user_agent: str
    data_dir: Path
    cache_dir: Path
    games_db_path: Path
    games_embeddings_path: Path
    price_history_path: Path
    games_cache_ttl_hours: int
    stores_cache_ttl_hours: int
    request_timeout_seconds: int
    chroma_persist_dir: str | None


def load_config() -> AppConfig:
    load_dotenv()

    data_dir = BASE_DIR / "data"
    cache_dir = data_dir / "cache"
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "deal-hunter"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:latest"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        cheapshark_base_url=os.getenv("CHEAPSHARK_BASE_URL", "https://www.cheapshark.com/api/1.0"),
        cheapshark_user_agent=os.getenv(
            "CHEAPSHARK_USER_AGENT",
            "DealHunter/1.0 (contact@example.com)",
        ),
        data_dir=data_dir,
        cache_dir=cache_dir,
        games_db_path=Path(os.getenv("GAMES_DB_PATH", str(data_dir / "games_database.csv"))),
        games_embeddings_path=Path(
            os.getenv("GAMES_EMBEDDINGS_PATH", str(data_dir / "games_with_embeddings.csv"))
        ),
        price_history_path=Path(
            os.getenv("PRICE_HISTORY_PATH", str(data_dir / "price_history.csv"))
        ),
        games_cache_ttl_hours=int(os.getenv("GAMES_CACHE_TTL_HOURS", "24")),
        stores_cache_ttl_hours=int(os.getenv("STORES_CACHE_TTL_HOURS", "24")),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR"),
    )
