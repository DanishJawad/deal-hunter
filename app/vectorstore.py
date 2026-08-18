from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except Exception:  # pragma: no cover - optional dependency
    chromadb = None

from .config import AppConfig
from .error_handler import GameDataError, VectorStoreError
from .models import Game
from .ollama_helper import embed_text

LOGGER = logging.getLogger(__name__)


class ChromaStore:
    def __init__(self, config: AppConfig, collection_name: str | None = None) -> None:
        if chromadb is None:
            raise VectorStoreError("chromadb package not installed")
        self.config = config
        if config.chroma_persist_dir:
            settings = ChromaSettings(persist_directory=config.chroma_persist_dir, is_persistent=True)
            try:
                from pathlib import Path

                Path(config.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
                self.client = chromadb.Client(settings=settings)
            except ValueError:
                LOGGER.warning("Chroma settings rejected; attempting to backup legacy persist directory and retry...")
                from pathlib import Path
                import shutil, time

                p = Path(config.chroma_persist_dir)
                if p.exists() and any(p.iterdir()):
                    backup = p.parent / f"{p.name}.legacy_backup_{int(time.time())}"
                    LOGGER.info("Backing up existing persist directory to %s", backup)
                    shutil.move(str(p), str(backup))
                try:
                    self.client = chromadb.Client(settings=settings)
                except Exception:
                    LOGGER.warning("Failed to create persistent Chroma client after backup; falling back to in-memory client.")
                    self.client = chromadb.Client()
        else:
            self.client = chromadb.Client()
        self.collection_name = collection_name or config.chroma_collection_name
        # create or get existing collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=self.collection_name)

    def get_vector_count(self) -> int:
        try:
            # chroma has count() in newer versions
            if hasattr(self.collection, "count"):
                return int(self.collection.count())
            # fallback: fetch ids (may be expensive)
            results = self.collection.get(include=["ids"]) or {}
            ids = results.get("ids") or []
            return len(ids)
        except Exception:
            return 0

    def ensure_games_indexed(self, embeddings_path: Path, min_count: int = 150) -> int:
        current = self.get_vector_count()
        if current >= min_count:
            return current
        self.upsert_games_from_embeddings(embeddings_path)
        return self.get_vector_count()

    def upsert_games_from_embeddings(self, embeddings_path: Path, batch_size: int = 100) -> None:
        if not embeddings_path.exists():
            raise GameDataError(f"Missing embeddings file at {embeddings_path}")

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict] = []

        with embeddings_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                embedding = _parse_embedding(row.get("embedding"))
                if not embedding:
                    continue
                game_id = (row.get("game_id") or row.get("id") or row.get("title") or "").strip()
                if not game_id:
                    continue
                metadata = {
                    "title": (row.get("title") or "").strip(),
                    "genres": _split_list(row.get("genres")),
                    "description": (row.get("description") or "").strip(),
                    "tags": _split_list(row.get("tags")),
                    "aliases": _split_list(row.get("aliases")),
                    "typical_price": _to_optional_float(row.get("typical_price")),
                    "metacritic_score": _to_optional_float(row.get("metacritic_avg")),
                }
                ids.append(game_id)
                embeddings.append(embedding)
                metadatas.append(metadata)

                if len(ids) >= batch_size:
                    self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
                    ids = []
                    embeddings = []
                    metadatas = []

        if ids:
            self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search_similar_games(self, query: str, top_k: int = 5, genres: list[str] | None = None) -> list[Game]:
        vector = embed_text(query)
        # Chroma's `where` filter only matches scalar metadata fields; "genres" is
        # stored as a list per game, so genre filtering is done client-side on the
        # over-fetched candidate set instead of pushed down to the query.
        fetch_k = top_k * 4 if genres else top_k
        try:
            results = self.collection.query(
                query_embeddings=[vector],
                n_results=fetch_k,
                include=["metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError("Chroma query failed") from exc

        matches = []
        raw_ids = results.get("ids") or []
        raw_metadatas = results.get("metadatas") or []

        ids = raw_ids[0] if raw_ids and isinstance(raw_ids[0], list) else raw_ids
        metadatas = raw_metadatas[0] if raw_metadatas and isinstance(raw_metadatas[0], list) else raw_metadatas

        wanted_genres = {genre.lower() for genre in genres} if genres else None

        for idx, _id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
            game_genres = list(metadata.get("genres") or [])
            if wanted_genres and not ({g.lower() for g in game_genres} & wanted_genres):
                continue
            matches.append(
                Game(
                    game_id=_id,
                    title=metadata.get("title") or "",
                    genres=game_genres,
                    description=metadata.get("description"),
                    tags=list(metadata.get("tags") or []),
                    aliases=list(metadata.get("aliases") or []),
                    typical_price=metadata.get("typical_price"),
                    metacritic_score=metadata.get("metacritic_score"),
                )
            )
            if len(matches) >= top_k:
                break
        return matches


def get_vector_store(config: AppConfig) -> ChromaStore:
    return ChromaStore(config)


def _parse_embedding(raw: str | None) -> list[float]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [float(value) for value in payload]


def _split_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _to_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
