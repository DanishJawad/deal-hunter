from __future__ import annotations

import base64
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st

from .agent import build_agent_executor, build_llm, build_tools, get_recommendations, run_deal_agent_sync
from .cheapshark import CheapSharkClient
from .config import AppConfig, load_config
from .error_handler import AppError, friendly_error_message
from .games_db import load_games_database
from .logging_config import configure_logging
from .ollama_helper import init_ollama
from .preferences import extract_preferences_from_query, get_preferences, merge_preferences, save_preferences
from .vectorstore import get_vector_store

LOGGER = logging.getLogger(__name__)


def _load_logo_data_uri(path: str) -> str:
    logo_path = Path(path)
    if not logo_path.exists():
        return ""
    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@st.cache_resource
def get_config() -> AppConfig:
    return load_config()


@st.cache_resource
def get_cheapshark_client() -> CheapSharkClient:
    config = load_config()
    return CheapSharkClient(
        base_url=config.cheapshark_base_url,
        timeout_seconds=config.request_timeout_seconds,
        cache_dir=config.cache_dir,
        games_cache_ttl_seconds=config.games_cache_ttl_hours * 3600,
        stores_cache_ttl_seconds=config.stores_cache_ttl_hours * 3600,
        user_agent=config.cheapshark_user_agent,
    )


@st.cache_resource
def get_vector_store_resource():
    config = load_config()
    return get_vector_store(config)


@st.cache_resource
def get_llm():
    config = load_config()
    return build_llm(model=config.ollama_chat_model, base_url=config.ollama_base_url)


@st.cache_data(ttl=3600)
def get_games_lookup():
    config = load_config()
    return load_games_database(config.games_db_path)


@st.cache_data(ttl=3600)
def get_store_map():
    client = get_cheapshark_client()
    return client.fetch_stores()


@st.cache_data(ttl=1800)
def cached_similar_games(query: str, top_k: int) -> list[dict]:
    store = get_vector_store_resource()
    return [game.model_dump() for game in store.search_similar_games(query=query, top_k=top_k)]


def run_app() -> None:
    configure_logging()
    st.set_page_config(page_title="Deal Hunter", page_icon="\U0001f3ae", layout="wide")

    st.markdown(
        """
        <style>
        /* Cache bust: 2026-06-01 */
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono&display=swap');
        :root {
            --neon-cyan: #19f5ff;
            --neon-pink: #ff2d95;
            --neon-lime: #b7ff3c;
            --ink: #f5f7ff;
            --muted: #9aa3b2;
            --surface: rgba(16, 20, 32, 0.82);
            --surface-strong: rgba(14, 18, 30, 0.92);
        }
        .stApp {
            background:
                radial-gradient(800px 420px at 10% -10%, rgba(25, 245, 255, 0.18), transparent 60%),
                radial-gradient(900px 500px at 90% -20%, rgba(255, 45, 149, 0.16), transparent 55%),
                linear-gradient(180deg, #0b0f1a 0%, #0a0e18 30%, #0b0f1a 100%);
            color: var(--ink);
        }
        .block-container {
            padding-top: 3rem;
            padding-bottom: 2.6rem;
            padding-left: 2.2rem;
            padding-right: 2.2rem;
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            font-family: "Space Grotesk", "Helvetica Neue", Arial, sans-serif;
        }
        .hero-shell {
            background: var(--surface-strong);
            border: 1px solid rgba(25, 245, 255, 0.2);
            border-radius: 28px;
            padding: 28px 32px;
            box-shadow: 0 30px 80px rgba(3, 8, 20, 0.7), 0 0 40px rgba(25, 245, 255, 0.08);
            backdrop-filter: blur(18px);
            position: relative;
            overflow: hidden;
        }
        .hero-shell::after {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(400px 120px at 20% 0%, rgba(25, 245, 255, 0.08), transparent 60%);
            pointer-events: none;
        }
        .hero-inner {
            display: flex;
            align-items: center;
            gap: 24px;
        }
        .hero-logo {
            width: 78px;
            height: 78px;
            object-fit: contain;
            filter: drop-shadow(0 0 18px rgba(25, 245, 255, 0.4));
        }
        .hero-title {
            font-size: 2.6rem;
            font-weight: 700;
            margin: 0 0 6px 0;
            letter-spacing: 0.02em;
            line-height: 1.05;
        }
        .hero-subtitle {
            color: var(--muted);
            font-size: 1.05rem;
            margin: 0;
            line-height: 1.5;
        }
        .hero-strip {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-top: 20px;
            position: relative;
            z-index: 1;
        }
        .hero-chip {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(25, 245, 255, 0.14);
            border-radius: 16px;
            padding: 14px 16px;
        }
        .hero-chip-title {
            display: block;
            color: var(--cyan, var(--neon-cyan));
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 5px;
        }
        .hero-chip-text {
            color: var(--text);
            font-size: 0.96rem;
            line-height: 1.4;
        }
        .form-shell {
            margin-top: 18px;
            background: rgba(16, 20, 32, 0.72);
            border: 1px solid rgba(25, 245, 255, 0.12);
            border-radius: 24px;
            padding: 18px 18px 10px;
            box-shadow: 0 20px 50px rgba(3, 7, 18, 0.38);
        }
        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin: 22px 0 12px;
        }
        .section-title-wrap {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .game-thumb {
            width: 84px;
            height: 84px;
            border-radius: 16px;
            object-fit: cover;
            border: 1px solid rgba(25, 245, 255, 0.18);
            box-shadow: 0 14px 30px rgba(3, 7, 18, 0.55);
        }
        .section-kicker {
            color: var(--muted);
            font-size: 0.88rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
        }
        .section-title {
            margin: 0;
            font-size: 1.3rem;
        }
        .results-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
            margin-top: 24px;
        }
        .deal-card {
            background: rgba(16, 20, 32, 0.78);
            border: 1px solid rgba(25, 245, 255, 0.12);
            border-radius: 20px;
            padding: 18px;
            margin-top: 0 !important;
            margin-bottom: 16px;
        }
        .deal-card-top {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: flex-start;
            margin-bottom: 12px;
        }
        .deal-game {
            font-size: 1.02rem;
            font-weight: 700;
            margin: 0 0 4px;
            line-height: 1.3;
        }
        .deal-store {
            color: var(--muted);
            font-size: 0.92rem;
        }
        .deal-price {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--green);
            white-space: nowrap;
        }
        .deal-meta {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 12px;
        }
        .deal-meta span {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.06);
            font-size: 0.86rem;
        }
        .deal-link {
            display: inline-block !important;
            margin-top: 14px !important;
            padding: 10px 18px !important;
            border-radius: 12px !important;
            background: transparent !important;
            color: #ffffff !important;
            text-decoration: none !important;
            font-weight: 600 !important;
            font-size: 1.02rem !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            cursor: pointer !important;
            transition: background 0.2s, border-color 0.2s !important;
        }
        .deal-link:hover {
            background: rgba(255, 255, 255, 0.08) !important;
            border-color: rgba(255, 255, 255, 0.3) !important;
        }
        .deal-store-btn {
            display: inline-block !important;
            margin-top: 12px !important;
            padding: 10px 18px !important;
            border-radius: 12px !important;
            background: transparent !important;
            color: #ffffff !important;
            text-decoration: none !important;
            font-weight: 600 !important;
            font-size: 1.02rem !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            cursor: pointer !important;
            transition: background 0.2s, border-color 0.2s !important;
        }
        .deal-store-btn:hover {
            background: rgba(255, 255, 255, 0.08) !important;
            border-color: rgba(255, 255, 255, 0.3) !important;
        }
        .deal-card .small {
            line-height: 1.45;
        }
        .stTextInput label {
            margin-bottom: 8px;
        }
        .stTextInput input {
            border-radius: 16px;
            border: 1px solid rgba(25, 245, 255, 0.22);
            background: rgba(9, 12, 20, 0.8);
            color: var(--ink);
            padding: 14px 16px;
            font-size: 1rem;
            line-height: 1.2;
            box-shadow: 0 12px 30px rgba(3, 7, 18, 0.65);
        }
        .stTextInput input::placeholder {
            color: #7d8798;
        }
        .stSlider [data-baseweb="slider"] > div {
            color: var(--neon-cyan);
        }
        .stButton > button {
            border-radius: 16px;
            border: none;
            background: linear-gradient(135deg, #19f5ff 0%, #3a74ff 50%, #ff2d95 100%);
            color: #03131e;
            font-weight: 600;
            padding: 12px 20px;
            box-shadow: 0 14px 30px rgba(25, 245, 255, 0.2), 0 0 22px rgba(255, 45, 149, 0.15);
        }
        .stButton > button:hover {
            filter: brightness(1.02);
        }
        div[data-testid="stForm"] {
            margin-top: 20px;
            background: rgba(16, 20, 32, 0.72);
            border: 1px solid rgba(25, 245, 255, 0.12);
            border-radius: 24px;
            padding: 22px 22px 16px;
            box-shadow: 0 20px 50px rgba(3, 7, 18, 0.38);
            position: relative;
            z-index: 1;
        }
        div[data-testid="stFormSubmitButton"] button {
            width: auto;
        }
        .stDataFrame, .stMetric, .stMarkdown, .stAlert {
            background: var(--surface);
            border-radius: 18px;
            border: 1px solid rgba(25, 245, 255, 0.15);
            box-shadow: 0 16px 40px rgba(3, 7, 18, 0.55);
        }
        div[data-testid="metric-container"] {
            padding: 14px 16px !important;
            text-align: center !important;
        }
        div[data-testid="metric-container"] > div {
            text-align: center !important;
        }
        div[data-testid="metric-container"] > div > div {
            text-align: center !important;
        }
        div[data-testid="stAlert"] > div {
            padding: 12px 16px;
        }
        .stMarkdown table {
            background: transparent;
            color: var(--ink);
        }
        .stMarkdown table a {
            color: var(--neon-cyan);
            text-decoration: none;
        }
        .stMarkdown table a:hover {
            text-decoration: underline;
        }
        .stProgress > div > div {
            background: linear-gradient(90deg, #19f5ff 0%, #b7ff3c 100%);
        }
        @keyframes pulseGlow {
            0% { box-shadow: 0 0 14px rgba(25, 245, 255, 0.12); }
            50% { box-shadow: 0 0 26px rgba(255, 45, 149, 0.18); }
            100% { box-shadow: 0 0 14px rgba(25, 245, 255, 0.12); }
        }
        .hero-shell {
            animation: pulseGlow 6s ease-in-out infinite;
        }
        @media (max-width: 900px) {
            .block-container {
                padding-top: 2rem;
                padding-left: 1.2rem;
                padding-right: 1.2rem;
            }
            .hero-inner {
                flex-direction: column;
                align-items: flex-start;
            }
            .hero-strip {
                grid-template-columns: 1fr;
            }
            .results-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    logo_uri = _load_logo_data_uri("assets/dealhunter_logo.png")
    logo_html = ""
    if logo_uri:
        logo_html = f'<img class="hero-logo" src="{logo_uri}" alt="Deal Hunter logo" />'
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-inner">
                {logo_html}
                <div>
                    <div class="hero-title">Deal Hunter</div>
                    <p class="hero-subtitle">Get the best PC game price in seconds.</p>
                </div>
            </div>
            <div class="hero-strip">
                <div class="hero-chip">
                    <span class="hero-chip-title">Best for</span>
                    <div class="hero-chip-text">Exact game prices and live store comparisons.</div>
                </div>
                <div class="hero-chip">
                    <span class="hero-chip-title">Also does</span>
                    <div class="hero-chip-text">Similar-game discovery when you want ideas.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    config = get_config()

    with st.form("search_form"):
        col_query, col_action = st.columns([4, 1])
        with col_query:
            query = st.text_input(
                "Search game or ask for similar games",
                placeholder="Where can I get GTA V the cheapest?",
            )
        with col_action:
            st.write(" ")
            submitted = st.form_submit_button("Find Deals")

    if not submitted:
        backend = (config.vectorstore_backend or "pinecone").lower()
        backend_label = "Chroma" if backend == "chroma" else "Pinecone"
        st.caption(f"Using local Ollama model + {backend_label}")
        return

    if not query.strip():
        st.warning("Please enter a game query to search.")
        return

    try:
        with st.spinner("Finding deals with Ollama..."):
            init_ollama(config)
            lookup = get_games_lookup()
            available_genres = sorted(
                {genre for game in lookup.games for genre in game.genres if genre}
            )
            cheapshark = get_cheapshark_client()
            try:
                store = get_vector_store_resource()
            except AppError as exc:
                store = None
                st.warning(friendly_error_message(exc))
            store_map = get_store_map()

            if store is not None:
                try:
                    store.ensure_games_indexed(config.games_embeddings_path)
                except AppError as exc:
                    st.warning(friendly_error_message(exc))

            prefs = get_preferences()
            extracted = extract_preferences_from_query(
                query,
                available_genres=available_genres,
                default_budget=float(prefs.budget),
                default_genres=prefs.favorite_genres,
            )
            prefs = merge_preferences(prefs, extracted)
            save_preferences(prefs)

            results = get_recommendations(
                query=query.strip(),
                max_price=float(prefs.budget),
                lookup=lookup,
                cheapshark=cheapshark,
                store=store,
                store_map=store_map,
                price_history_path=config.price_history_path,
                favorite_genres=prefs.favorite_genres,
                limit=5,
                search_fn=cached_similar_games if store is not None else None,
            )
            llm = get_llm()
            tools = build_tools(cheapshark, store, lookup, store_map)
            executor = build_agent_executor(llm, tools)
            agent_summary = run_deal_agent_sync(query, executor)

    except AppError as exc:
        LOGGER.exception("Search failed")
        st.error(friendly_error_message(exc))
        return
    except Exception as exc:  # pragma: no cover - unexpected errors
        LOGGER.exception("Search failed")
        st.error(f"Search failed: {exc}")
        return

    if results.suggestions:
        st.info("Did you mean: " + ", ".join(results.suggestions))

    if not results.recommendations:
        st.info("No deals found. Try a different game or higher max price.")
        backend = (config.vectorstore_backend or "pinecone").lower()
        backend_label = "Chroma" if backend == "chroma" else "Pinecone"
        st.caption(f"Using local Ollama model + {backend_label}")
        return

    if results.is_exact_game:
        best_rec = min(results.recommendations, key=lambda rec: rec.deal.sale_price)
        best_url = f"https://www.cheapshark.com/redirect?dealID={best_rec.deal.deal_id}"
        thumb_url = best_rec.deal.thumb or ""
        thumb_html = ""
        if thumb_url:
            thumb_html = f'<img class="game-thumb" src="{thumb_url}" alt="{best_rec.game.title} cover" />'
        st.markdown(
            f"""
            <div class="section-header">
                <div class="section-title-wrap">
                    {thumb_html}
                    <div>
                        <div class="section-kicker">Exact game pricing</div>
                        <h2 class="section-title">{best_rec.game.title}</h2>
                    </div>
                </div>
                    <a class="deal-link" href="{best_url}" target="_blank">Open store</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        best_cols = st.columns(3)
        best_cols[0].metric("Best price", f"${best_rec.deal.sale_price:.2f}")
        best_cols[1].metric("Store", best_rec.store_name or "Unknown")
        best_cols[2].metric("Discount", f"{best_rec.discount_percent or 0:.0f}%")

        st.markdown("<div class='results-grid'>", unsafe_allow_html=True)
        rows = sorted(results.recommendations, key=lambda rec: rec.deal.sale_price)
        rows = [rec for rec in rows if rec.deal.deal_id != best_rec.deal.deal_id]
        seen_stores: set[str] = set()
        for rec in rows:
            store_key = (rec.store_name or "unknown").strip().lower()
            if store_key in seen_stores:
                continue
            seen_stores.add(store_key)
            link = f"https://www.cheapshark.com/redirect?dealID={rec.deal.deal_id}"
            st.markdown(
                f"""
                <div class="deal-card">
                    <div class="deal-card-top">
                        <div>
                            <div class="deal-game">{rec.store_name or 'Unknown'}</div>
                            <div class="deal-store">Was ${rec.deal.normal_price:.2f}</div>
                        </div>
                        <div class="deal-price">${rec.deal.sale_price:.2f}</div>
                    </div>
                    <div class="deal-meta">
                        <span>Savings {rec.discount_percent or 0:.0f}%</span>
                        <span>Rating {rec.deal.deal_rating:.1f}</span>
                    </div>
                    <a class="deal-store-btn" href="{link}" target="_blank">Open store</a>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="section-header">
                <div>
                    <div class="section-kicker">Similar games</div>
                    <h2 class="section-title">Best matches and current deals</h2>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for rec in results.recommendations:
            deal_score = rec.deal_evaluation.score if rec.deal_evaluation else 0
            st.markdown(
                f"""
                <div class="deal-card" style="margin-top: 12px;">
                    <div class="deal-card-top">
                        <div>
                            <div class="deal-game">{rec.game.title}</div>
                            <div class="deal-store">{rec.store_name or 'Unknown'} · Metacritic {rec.game.metacritic_score or 'N/A'}</div>
                        </div>
                        <div class="deal-price">${rec.deal.sale_price:.2f}</div>
                    </div>
                    <div class="deal-meta">
                        <span>Was ${rec.deal.normal_price:.2f}</span>
                        <span>Savings {rec.discount_percent or 0:.0f}%</span>
                        <span>Score {deal_score:.0f}</span>
                    </div>
                    <div class="small" style="margin-top: 10px; color: var(--muted);">{rec.reasoning or 'Matched from your query.'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("Why these? (Agent reasoning)"):
        st.write(agent_summary or "No reasoning available.")

    st.caption("All processing done locally with Ollama + Pinecone (zero API costs)")
