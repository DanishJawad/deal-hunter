from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppConfig:
    pinecone_api_key: str
    vectorstore_backend: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embed_model: str
    ollama_embed_dim: int
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
    chroma_server_url: str | None

    def validate(self) -> list[str]:
        missing = []
        backend = (self.vectorstore_backend or "").lower()
        if backend != "chroma":
            if not self.pinecone_api_key:
                missing.append("PINECONE_API_KEY")
            if not self.pinecone_index_name:
                missing.append("PINECONE_INDEX_NAME")
            if not self.pinecone_cloud:
                missing.append("PINECONE_CLOUD")
            if not self.pinecone_region:
                missing.append("PINECONE_REGION")
        return missing


def load_config() -> AppConfig:
    load_dotenv()

    data_dir = BASE_DIR / "data"
    cache_dir = data_dir / "cache"
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        vectorstore_backend=os.getenv("VECTORSTORE_BACKEND", "chroma"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "deal-hunter"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:latest"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        ollama_embed_dim=int(os.getenv("OLLAMA_EMBED_DIM", "768")),
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
        chroma_server_url=os.getenv("CHROMA_SERVER_URL"),
    )
