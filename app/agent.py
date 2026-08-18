from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from langchain_ollama import ChatOllama

from .cheapshark import CheapSharkClient
from .deal_logic import build_recommendation, personalize_recommendations
from .error_handler import VectorStoreError
from .games_db import (
    GameLookup,
    find_game_by_name,
    find_game_in_query,
    normalize_title,
    search_by_keyword,
    suggest_similar_titles,
)
from .models import Game, Recommendation
from .nlp import extract_price_limit
from .prompts import AGENT_SYSTEM_PROMPT
from .vectorstore import ChromaStore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DealResults:
    recommendations: list[Recommendation]
    suggestions: list[str]
    is_exact_game: bool = False


def build_llm(model: str, base_url: str) -> ChatOllama:
    return ChatOllama(model=model, base_url=base_url, temperature=0.2)


def generate_result_reasoning(
    llm: ChatOllama, query: str, recommendations: list[Recommendation]
) -> str:
    """Explain the already-retrieved results shown to the user.

    The LLM only ever sees the exact candidate list that's already being
    rendered in the result cards, so it can't introduce a game that wasn't
    actually retrieved. This is a plain explanation call, not a tool-calling
    loop — there's nothing left to search for at this point.
    """
    if not recommendations:
        return "No matching deals were found for this query."

    candidate_lines = []
    for rec in recommendations:
        discount = rec.discount_percent or 0
        verdict = rec.deal_evaluation.verdict if rec.deal_evaluation else "N/A"
        candidate_lines.append(
            f"- {rec.game.title}: ${rec.deal.sale_price:.2f} at {rec.store_name or 'Unknown'} "
            f"({discount:.0f}% off, was ${rec.deal.normal_price:.2f}) - {verdict}"
        )
    candidates_text = "\n".join(candidate_lines)

    system_instructions = (
        AGENT_SYSTEM_PROMPT
        + "\n\nThe search has already run. Below is the exact, final list of games and deals "
        + "being shown to the user right now. Only explain why these specific results match "
        + "the query. Do not introduce, suggest, or mention any other game title, even ones "
        + "you know are similar - if it's not in the list below, it doesn't exist for this answer."
    )
    user_message = f"User query: {query}\n\nRetrieved results:\n{candidates_text}"

    try:
        resp = llm.invoke([("system", system_instructions), ("user", user_message)])
        content = getattr(resp, "content", str(resp)) or ""
        return content.strip() or "No reasoning available."
    except Exception as exc:  # pragma: no cover - runtime errors
        LOGGER.warning("Reasoning generation failed", extra={"error": str(exc)})
        return "Unable to generate reasoning at this time."


def _needs_recommendations(query: str) -> bool:
    lowered = query.lower()
    triggers = ["like", "similar", "recommend", "recommendation", "games", "titles"]
    return any(token in lowered for token in triggers)


def get_recommendations(
    *,
    query: str,
    max_price: float,
    lookup: GameLookup,
    cheapshark: CheapSharkClient,
    store: ChromaStore | None,
    store_map: dict[str, str],
    price_history_path,
    favorite_genres: list[str],
    limit: int = 5,
    search_fn: Callable[[str, int], list[dict]] | None = None,
) -> DealResults:
    alias_match = find_game_by_name(query, lookup) or find_game_in_query(query, lookup)
    recommend_mode = _needs_recommendations(query)
    is_exact_game = alias_match is not None and not recommend_mode
    price_limit = extract_price_limit(query)
    effective_max_price = price_limit if price_limit is not None else max_price

    if is_exact_game:
        deals = cheapshark.fetch_deals(query=alias_match.title, max_price=price_limit)
        target_norm = normalize_title(alias_match.title)
        filtered_deals = [
            deal for deal in deals if target_norm and target_norm in normalize_title(deal.title)
        ]
        if filtered_deals:
            deals = filtered_deals
        if not deals:
            suggestions = suggest_similar_titles(query, lookup)
            return DealResults(recommendations=[], suggestions=suggestions, is_exact_game=True)

        recommendations: list[Recommendation] = []
        for deal in deals:
            store_name = store_map.get(deal.store_id)
            recommendations.append(
                build_recommendation(
                    game=alias_match,
                    deal=deal,
                    store_name=store_name,
                    price_history_path=price_history_path,
                )
            )
        return DealResults(recommendations=recommendations, suggestions=[], is_exact_game=True)

    games: list[Game] = []
    if alias_match and not recommend_mode:
        games = [alias_match]
    else:
        if store is None:
            games = search_by_keyword(query, lookup, limit=limit)
        else:
            try:
                if search_fn:
                    games = [Game.model_validate(item) for item in search_fn(query, limit)]
                else:
                    games = store.search_similar_games(
                        query=alias_match.title if alias_match else query,
                        top_k=limit,
                    )
            except VectorStoreError:
                games = search_by_keyword(query, lookup, limit=limit)

    suggestions: list[str] = []
    if not games:
        suggestions = suggest_similar_titles(query, lookup)
        return DealResults(recommendations=[], suggestions=suggestions)

    recommendations: list[Recommendation] = []
    for game in games:
        deals = cheapshark.fetch_deals(query=game.title, max_price=effective_max_price)
        if not deals:
            continue
        best_deal = max(deals, key=lambda item: item.deal_rating)
        store_name = store_map.get(best_deal.store_id)
        recommendations.append(
            build_recommendation(
                game=game,
                deal=best_deal,
                store_name=store_name,
                price_history_path=price_history_path,
            )
        )

    if not recommendations:
        fallback_deals = cheapshark.fetch_deals(query=query, max_price=effective_max_price)
        for deal in fallback_deals[:limit]:
            fallback_game = Game(title=deal.title)
            store_name = store_map.get(deal.store_id)
            recommendations.append(
                build_recommendation(
                    game=fallback_game,
                    deal=deal,
                    store_name=store_name,
                    price_history_path=price_history_path,
                )
            )

    recommendations = personalize_recommendations(recommendations, favorite_genres)
    return DealResults(recommendations=recommendations, suggestions=suggestions, is_exact_game=False)
