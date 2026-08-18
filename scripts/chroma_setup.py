from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import load_config
from app.ollama_helper import embed_text, init_ollama

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except Exception:
    chromadb = None


def init_chroma(persist_dir: str | None):
    if chromadb is None:
        raise SystemExit("chromadb package not installed; pip install chromadb")
    if persist_dir:
        # Use the new Settings fields and avoid legacy chroma_db_impl values
        settings = ChromaSettings(persist_directory=persist_dir, is_persistent=True)
        try:
            # ensure directory exists before creating client
            from pathlib import Path
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            return chromadb.Client(settings=settings)
        except ValueError as exc:
            print("Warning: chromadb settings rejected; attempting to backup legacy persist directory and retry...")
            import traceback
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            from pathlib import Path
            import shutil, time

            p = Path(persist_dir)
            if p.exists() and any(p.iterdir()):
                backup = p.parent / f"{p.name}.legacy_backup_{int(time.time())}"
                print(f"Backing up existing persist directory to {backup}")
                shutil.move(str(p), str(backup))
            try:
                # retry after backup (directory will be created by Chroma)
                return chromadb.Client(settings=settings)
            except Exception as exc2:
                import traceback
                print("Failed to create persistent Chroma client after backup; exception:")
                traceback.print_exception(type(exc2), exc2, exc2.__traceback__)
                print("Falling back to in-memory client. If you need to migrate data, install chroma-migrate and run it.")
                return chromadb.Client()
    return chromadb.Client()


def create_game_collection(client, name: str):
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name=name)


def embed_and_upsert_games(games_df: pd.DataFrame, collection, batch_size: int = 100) -> None:
    ids = []
    embeddings = []
    metadatas = []

    for _, row in games_df.iterrows():
        embedding = row.get("embedding")
        if isinstance(embedding, str):
            try:
                embedding = json.loads(embedding)
            except json.JSONDecodeError:
                continue
        if not isinstance(embedding, list):
            continue
        game_id = str(row.get("game_id"))
        metadata = {
            "title": row.get("title"),
            "genres": str(row.get("genres", "")).split(","),
            "description": row.get("description"),
            "tags": str(row.get("tags", "")).split(","),
            "aliases": str(row.get("aliases", "")).split(","),
            "typical_price": row.get("typical_price"),
            "metacritic_score": row.get("metacritic_avg"),
        }
        ids.append(game_id)
        embeddings.append(embedding)
        metadatas.append(metadata)

        if len(ids) >= batch_size:
            collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
            ids = []
            embeddings = []
            metadatas = []

    if ids:
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)


def search_similar_games(collection, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    vector = embed_text(query)
    results = collection.query(query_embeddings=[vector], n_results=top_k, include=["metadatas", "distances"]) 
    return results.get("ids") or []


def main() -> None:
    config = load_config()
    init_ollama()

    client = init_chroma(config.chroma_persist_dir)
    collection = create_game_collection(client, config.chroma_collection_name)

    embeddings_path = Path(config.games_embeddings_path)
    if not embeddings_path.exists():
        raise SystemExit(
            f"Missing embeddings at {embeddings_path}. Run generate_game_embeddings.py first."
        )

    games_df = pd.read_csv(embeddings_path)
    embed_and_upsert_games(games_df, collection)
    print("Upsert complete")

    matches = search_similar_games(collection, "Games like GTA V", top_k=3)
    print(matches)


if __name__ == "__main__":
    main()
