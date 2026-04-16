from __future__ import annotations
import html
import random
import re
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.config import load_config
    from src.pipeline import (
    attach_verifications,
    build_answer_clients,
    build_judge_client,
    build_verifier_client,
    generate_answer,
    load_questions,
    run_gatekeeper,
    run_judge,
    run_verifier,
    resolve_tie_break,
    )
except Exception as e:
    st.error("Could not import project modules.")
    st.exception(e)
    st.stop()


APP_TITLE = "Cognitive Court"
DEFAULT_QUESTIONS_PATH = "data/splits/dev.json"
MODE_OPTIONS = ["standard", "cot", "both"]
MODEL_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_REJECT_REPLY = (
    "I cannot help with that question, but I can help with a neutral scientific question."
)


def normalize_mode(value: str) -> str:
    value = (value or "").strip().lower()
    return value if value in MODE_OPTIONS else "both"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Mono:wght@400;500;700&family=Syne:wght@400;600;700;800&display=swap');

        *, *::before, *::after { box-sizing: border-box; }

        .stApp {
            background: #080c14;
            color: #dce8f5;
            font-family: 'Syne', sans-serif;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
            border: none !important;
            height: 0rem !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #08111f 0%, #0b1730 45%, #08101c 100%) !important;
            border-right: 1px solid rgba(56,189,248,0.14);
            box-shadow: 6px 0 26px rgba(0,0,0,0.35);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 0.35rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        section[data-testid="stSidebar"] * {
            color: #e5eefb !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {
            color: #f8fbff !important;
            letter-spacing: 0.01em;
        }

        section[data-testid="stSidebar"] h2 {
            margin-top: 0.1rem !important;
            margin-bottom: 0.4rem !important;
        }

        section[data-testid="stSidebar"] .stMarkdown {
            margin-top: 0 !important;
            margin-bottom: 0.2rem !important;
        }

        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] label {
            color: #9db4d0 !important;
            line-height: 1.45;
        }

        section[data-testid="stSidebar"] .stRadio,
        section[data-testid="stSidebar"] .stToggle,
        section[data-testid="stSidebar"] .stSelectbox {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(148,163,184,0.16);
            border-radius: 16px;
            padding: 0.9rem 0.95rem;
            margin-top: 0.15rem;
            margin-bottom: 0.65rem;
            backdrop-filter: blur(6px);
        }

        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] .stToggle label,
        section[data-testid="stSidebar"] .stSelectbox label {
            color: #edf4ff !important;
            font-weight: 700 !important;
            font-size: 0.92rem !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="radio"] {
            gap: 0.35rem;
        }

        section[data-testid="stSidebar"] [role="radiogroup"] > label {
            padding: 0.25rem 0;
        }

        section[data-testid="stSidebar"] [data-baseweb="toggle"] {
            transform: scale(1.02);
        }

        section[data-testid="stSidebar"] input[type="text"],
        section[data-testid="stSidebar"] textarea {
            background: #0a1220 !important;
            color: #eef6ff !important;
            border: 1px solid rgba(148,163,184,0.22) !important;
            border-radius: 12px !important;
            caret-color: #7dd3fc !important;
        }

        section[data-testid="stSidebar"] input::placeholder {
            color: #64748b !important;
        }

        section[data-testid="stSidebar"] button {
            color: #e5eefb !important;
        }

        section[data-testid="stSidebar"] input:focus,
        section[data-testid="stSidebar"] textarea:focus {
            border-color: rgba(56,189,248,0.55) !important;
            box-shadow: 0 0 0 3px rgba(56,189,248,0.10) !important;
        }

        [data-testid="collapsedControl"] {
            background: rgba(15,23,42,0.9) !important;
            border: 1px solid rgba(56,189,248,0.18) !important;
            border-radius: 999px !important;
            color: #e5eefb !important;
        }

        .cc-sidebar-card {
            background: linear-gradient(135deg, rgba(56,189,248,0.08), rgba(168,85,247,0.06));
            border: 1px solid rgba(56,189,248,0.18);
            border-radius: 16px;
            padding: 0.85rem 1rem 0.8rem 1rem;
            margin: 0.45rem 0 0.6rem 0;
        }
        .cc-sidebar-card-title {
            font-family: 'DM Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #7dd3fc;
            margin-bottom: 0.45rem;
        }
        .cc-sidebar-card-body {
            color: #d7e7f7 !important;
            font-size: 0.92rem;
            line-height: 1.35;
        }

        section.main > div.block-container {
            max-width: 960px !important;
            padding-top: 0 !important;
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
            padding-bottom: 3rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        div[data-testid="stAppViewContainer"] .main {
            padding-left: 0 !important;
            padding-right: 0 !important;
            padding-top: 0 !important;
            margin-top: 0 !important;
        }

        section[data-testid="stMain"] {
            padding-top: 0 !important;
        }

        .cc-hero {
            position: relative;
            overflow: hidden;
            width: 100%;
            margin: -6.2rem 0 0 0;
            padding: 0.20rem 2rem 0.8rem 2rem;
            background: linear-gradient(135deg, #060b14 0%, #0c1526 60%, #0a1020 100%);
            border-bottom: 1px solid rgba(255,255,255,0.07);
        }
        .cc-hero::before {
            content: '';
            position: absolute;
            inset: 0;
            background:
                radial-gradient(ellipse 60% 80% at 10% 0%, rgba(56,189,248,0.12) 0%, transparent 60%),
                radial-gradient(ellipse 50% 60% at 90% 100%, rgba(168,85,247,0.10) 0%, transparent 55%);
            pointer-events: none;
        }
        .cc-hero-inner {
            position: relative;
            z-index: 1;
            display: flex;
            align-items: flex-end;
            gap: 2rem;
            justify-content: space-between;
            width: 100%;
        }
        .cc-wordmark { flex: 0 0 auto; }
        .cc-wordmark-top {
            font-family: 'Instrument Serif', Georgia, serif;
            font-style: italic;
            font-size: 4.3rem;
            line-height: 0.88;
            letter-spacing: -0.04em;
            color: #f0f6ff;
            display: block;
        }
        .cc-wordmark-bottom {
            font-family: 'Syne', sans-serif;
            font-weight: 800;
            font-size: 1.02rem;
            letter-spacing: 0.34em;
            text-transform: uppercase;
            color: #38bdf8;
            display: block;
            margin-top: 0.2rem;
            padding-left: 0.1rem;
        }
        .cc-hero-divider {
            width: 1px;
            height: 68px;
            flex: 0 0 auto;
            background: linear-gradient(to bottom, transparent, rgba(255,255,255,0.18), transparent);
        }
        .cc-hero-meta {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
            align-items: flex-start;
        }
        .cc-tagline {
            font-family: 'Instrument Serif', serif;
            font-size: 1.25rem;
            color: #94b8d8;
            font-style: italic;
            line-height: 1.3;
        }
        .cc-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.3rem;
        }
        .cc-pill {
            font-family: 'DM Mono', monospace;
            font-size: 0.68rem;
            font-weight: 500;
            letter-spacing: 0.06em;
            padding: 0.26rem 0.65rem;
            border: 1px solid rgba(56,189,248,0.25);
            border-radius: 4px;
            color: #7dd3fc;
            background: rgba(56,189,248,0.05);
        }

        .cc-main {
            padding: 0.35rem 0 4rem 0;
            max-width: 100%;
            margin: 0 auto;
        }

        .cc-section-heading {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.65rem;
            margin-top: 1rem;
        }
        .cc-question-helper {
            margin: 0 0 0.8rem 0;
            color: #9db4d0;
            font-size: 0.95rem;
            line-height: 1.4;
            font-family: 'Syne', sans-serif;
        }
        .cc-section-heading-line {
            flex: 1;
            height: 1px;
            background: linear-gradient(to right, rgba(56,189,248,0.3), transparent);
        }
        .cc-section-heading-text {
            font-family: 'DM Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #38bdf8;
            white-space: nowrap;
            margin-left: 0.15rem;
        }

        .stTextArea {
            margin-top: 0.1rem;
        }
        .stTextArea > div > div > textarea {
            background: rgba(15,25,45,0.9) !important;
            border: 1px solid rgba(56,189,248,0.2) !important;
            border-radius: 14px !important;
            color: #dce8f5 !important;
            font-family: 'Syne', sans-serif !important;
            font-size: 1.02rem !important;
            line-height: 1.7 !important;
            padding: 1.2rem 1.2rem !important;
        }
        .stTextArea > div > div > textarea:focus {
            border-color: rgba(56,189,248,0.55) !important;
            box-shadow: 0 0 0 3px rgba(56,189,248,0.08) !important;
        }
        .stTextArea textarea::placeholder {
            color: #7f97b6 !important;
            opacity: 1 !important;
        }
        .stTextArea label { display: none !important; }

        .stButton > button {
            border: 1px solid rgba(148,163,184,0.25) !important;
            border-radius: 10px !important;
            font-family: 'Syne', sans-serif !important;
            font-weight: 700 !important;
            font-size: 0.78rem !important;
            padding: 0.55rem 0.6rem !important;
            width: 100% !important;
            min-height: 2.7rem !important;
            background: rgba(224,242,254,0.96) !important;
            color: #0f172a !important;
            transition: all 0.2s ease !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            white-space: nowrap !important;
            text-align: center !important;
            line-height: 1 !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            background: #cfeefe !important;
        }

        div[data-testid="stHorizontalBlock"] > div:nth-of-type(2) .stButton > button {
            background: #e0f2fe !important;
        }
        div[data-testid="stHorizontalBlock"] > div:nth-of-type(2) .stButton > button:hover {
            background: #bae6fd !important;
            transform: translateY(-1px);
        }

        div[data-testid="stHorizontalBlock"] > div:nth-of-type(3) .stButton > button {
            background: #ede9fe !important;
        }
        div[data-testid="stHorizontalBlock"] > div:nth-of-type(3) .stButton > button:hover {
            background: #ddd6fe !important;
            transform: translateY(-1px);
        }

        div[data-testid="stHorizontalBlock"] > div:nth-of-type(4) .stButton > button {
            background: #fee2e2 !important;
        }
        div[data-testid="stHorizontalBlock"] > div:nth-of-type(4) .stButton > button:hover {
            background: #fecaca !important;
            transform: translateY(-1px);
        }

        .cc-question-card {
            background: rgba(56,189,248,0.04);
            border: 1px solid rgba(56,189,248,0.18);
            border-left: 3px solid #38bdf8;
            border-radius: 12px;
            padding: 1rem 1.15rem;
            margin-bottom: 1.4rem;
        }
        .cc-question-tag {
            font-family: 'DM Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #38bdf8;
            margin-bottom: 0.35rem;
        }
        .cc-question-text {
            font-family: 'Instrument Serif', serif;
            font-size: 1.2rem;
            color: #f0f6ff;
            line-height: 1.5;
        }

        .cc-best-banner {
            background: linear-gradient(135deg, rgba(251,191,36,0.08), rgba(251,191,36,0.03));
            border: 1px solid rgba(251,191,36,0.28);
            border-radius: 14px;
            padding: 1.05rem 1.3rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: flex-start;
            gap: 1rem;
        }
        .cc-best-icon {
            font-size: 1.9rem;
            line-height: 1;
            flex: 0 0 auto;
            margin-top: 0.05rem;
        }
        .cc-best-content { flex: 1; }
        .cc-best-title {
            font-family: 'DM Mono', monospace;
            font-size: 0.67rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #fcd34d;
            margin-bottom: 0.25rem;
        }
        .cc-result-line {
            font-family: 'Syne', sans-serif;
            font-weight: 800;
            font-size: 1.05rem;
            color: #fef3c7;
            margin-bottom: 0.35rem;
            display: flex;
            align-items: baseline;
            gap: 0.6rem;
            flex-wrap: wrap;
        }
        .cc-model-name {
            color: #f0f6ff;
        }
        .cc-inline-score {
            font-size: 0.83rem;
            color: #7dd3fc;
            font-weight: 400;
            white-space: nowrap;
        }
        .cc-best-answer {
            font-size: 0.98rem;
            color: #f8fbff;
            line-height: 1.7;
            white-space: pre-wrap;
            margin-top: 0.2rem;
        }
        .cc-best-reason {
            font-size: 0.87rem;
            color: #94a3b8;
            line-height: 1.55;
            margin-top: 0.75rem;
        }

        .cc-banner-score-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin: 0.55rem 0 0.7rem 0;
        }

        .cc-score-pill {
            font-family: 'DM Mono', monospace;
            font-size: 0.64rem;
            letter-spacing: 0.04em;
            color: #cdeeff;
            background: rgba(15,23,42,0.65);
            border: 1px solid rgba(125,211,252,0.16);
            border-radius: 999px;
            padding: 0.28rem 0.55rem;
            white-space: nowrap;
        }

        .cc-card {
            border: 1px solid rgba(148,163,184,0.13);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            background: rgba(12,22,40,0.75);
            margin-bottom: 1rem;
            overflow: visible;
        }
        .cc-card.best {
            border-color: rgba(251,191,36,0.45);
            background: rgba(251,191,36,0.025);
            box-shadow: 0 0 0 1px rgba(251,191,36,0.08), 0 6px 20px rgba(251,191,36,0.05);
        }
        .cc-card.winner {
            border-color: rgba(52,211,153,0.4);
            background: rgba(52,211,153,0.025);
        }
        .cc-card-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 0.6rem;
            margin-bottom: 0.7rem;
            flex-wrap: wrap;
        }
        .cc-card-model-wrap {
            display: flex;
            align-items: baseline;
            gap: 0.6rem;
            flex-wrap: wrap;
        }
        .cc-card-model {
            font-family: 'Syne', sans-serif;
            font-weight: 800;
            font-size: 0.95rem;
            color: #f0f6ff;
        }
        .cc-badge-row {
            display: flex;
            gap: 0.35rem;
            flex-wrap: wrap;
            align-items: center;
        }
        .cc-badge {
            font-family: 'DM Mono', monospace;
            font-size: 0.65rem;
            font-weight: 500;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            white-space: nowrap;
            line-height: 1.4;
        }
        .cc-badge.best-b  { background: rgba(251,191,36,0.14); color: #fcd34d; border: 1px solid rgba(251,191,36,0.28); }
        .cc-badge.win-b   { background: rgba(52,211,153,0.11); color: #6ee7b7; border: 1px solid rgba(52,211,153,0.22); }

        .cc-card-body {
            font-size: 0.91rem;
            line-height: 1.55;
            color: #b4cce0;
            white-space: pre-wrap;
            word-break: break-word;
            overflow: visible;
        }

        .cc-score-grid {
            display: flex;
            flex-wrap: nowrap;
            gap: 0.45rem;
            margin-top: 0.8rem;
            overflow-x: auto;
            padding-bottom: 0.15rem;
        }
        .cc-score-cell {
            flex: 1 1 0;
            min-width: 0;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 7px;
            padding: 0.42rem 0.5rem;
            text-align: center;
        }
        .cc-score-cell-label {
            font-family: 'DM Mono', monospace;
            font-size: 0.57rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.18rem;
        }
        .cc-score-cell-value {
            font-family: 'DM Mono', monospace;
            font-size: 0.95rem;
            font-weight: 500;
            color: #e2e8f0;
        }

        .cc-judge-card {
            background: rgba(12,22,40,0.8);
            border: 1px solid rgba(148,163,184,0.11);
            border-radius: 13px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.9rem;
        }
        .cc-judge-title {
            font-family: 'DM Mono', monospace;
            font-size: 0.67rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.65rem;
        }
        .cc-judge-reason {
            font-size: 0.87rem;
            color: #94a3b8;
            line-height: 1.55;
            padding-top: 0.5rem;
            border-top: 1px solid rgba(255,255,255,0.06);
        }

        .cc-divider {
            width: 1px;
            min-height: 100%;
            background: linear-gradient(to bottom, transparent, rgba(255,255,255,0.15), transparent);
            margin: 0 auto;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.3rem;
            background: rgba(255,255,255,0.03);
            padding: 0.3rem;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.07);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 7px;
            padding: 0.5rem 1rem;
            font-family: 'DM Mono', monospace;
            font-size: 0.76rem;
            letter-spacing: 0.06em;
            color: #64748b;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(14,165,233,0.85), rgba(139,92,246,0.8)) !important;
            color: white !important;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 10px;
            padding: 0.8rem;
        }

        .stInfo > div {
            background: rgba(56,189,248,0.06) !important;
            border: 1px solid rgba(56,189,248,0.18) !important;
            border-radius: 10px !important;
        }

        hr { border-color: rgba(255,255,255,0.07); }

        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: rgba(56,189,248,0.2);
            border-radius: 3px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hp(block: str) -> None:
    st.markdown(textwrap.dedent(block).strip(), unsafe_allow_html=True)


def clean_answer(text: str, mode: str = "standard") -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"\*(.*?)\*", r"\1", line)
        line = re.sub(r"_(.*?)_", r"\1", line)
        line = re.sub(r"[ \t]+", " ", line).strip()

        if mode == "cot" and re.fullmatch(r"\d+\.", line):
            continue

        if mode == "cot":
            line = re.sub(r"^\d+[\.\)]\s*", "", line)

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


@st.cache_data(show_spinner=False)
def cached_questions(path: str):
    try:
        return load_questions(path)
    except Exception:
        return []


@st.cache_resource(show_spinner=False)
def cached_config():
    return load_config()


@st.cache_resource(show_spinner=False)
def cached_answer_clients():
    return build_answer_clients(cached_config())


@st.cache_resource(show_spinner=False)
def cached_judge_client():
    return build_judge_client(cached_config())


@st.cache_resource(show_spinner=False)
def cached_verifier_client():
    return build_verifier_client(cached_config())


def modes_for_selection(selection: str) -> list[str]:
    selection = normalize_mode(selection)
    if selection == "both":
        return ["standard", "cot"]
    return [selection]


def _clamp_score(key: str, value: object) -> float:
    try:
        v = float(value or 0)
    except (TypeError, ValueError):
        v = 0.0

    if key == "correctness":
        return min(max(v, 0.0), 2.0)
    if key == "mechanism_coverage":
        return min(max(v, 0.0), 3.0)
    if key == "logical_coherence":
        return min(max(v, 0.0), 2.0)
    if key == "explanatory_depth":
        return min(max(v, 0.0), 3.0)
    if key in {"unsupported_claims_penalty", "verification_penalty"}:
        return min(max(v, 0.0), 2.0)
    return max(v, 0.0)


def render_hero() -> None:
    hp(
        """
        <div class="cc-hero">
            <div class="cc-hero-inner">
                <div class="cc-wordmark">
                    <span class="cc-wordmark-top">Cognitive</span>
                    <span class="cc-wordmark-top" style="color:#38bdf8;">Court</span>
                    <span class="cc-wordmark-bottom">Model Reasoning Arena</span>
                </div>
                <div class="cc-hero-divider"></div>
                <div class="cc-hero-meta">
                    <div class="cc-tagline">Where models generate scientific answers,<br>and evidence selects the best one.</div>
                    <div class="cc-pill-row">
                        <span class="cc-pill">Parallel generation</span>
                        <span class="cc-pill">Answer comparison judge</span>
                        <span class="cc-pill">Claim verification</span>
                        <span class="cc-pill">Best answer highlighted</span>
                    </div>
                </div>
            </div>
        </div>
        """
    )


def section_heading(text: str) -> None:
    hp(
        f"""
        <div class="cc-section-heading">
            <div class="cc-section-heading-text">{html.escape(text)}</div>
            <div class="cc-section-heading-line"></div>
        </div>
        """
    )


def render_question_card(question: str) -> None:
    hp(
        f"""
        <div class="cc-question-card">
            <div class="cc-question-tag">Question</div>
            <div class="cc-question-text">{html.escape(question)}</div>
        </div>
        """
    )


def score_fmt(score: dict, key: str) -> str:
    return f"{_clamp_score(key, score.get(key, 0)):.1f}"


def compute_total_score(score: dict) -> float:
    correctness = _clamp_score("correctness", score.get("correctness", 0))
    mechanism = _clamp_score("mechanism_coverage", score.get("mechanism_coverage", 0))
    coherence = _clamp_score("logical_coherence", score.get("logical_coherence", 0))
    depth = _clamp_score("explanatory_depth", score.get("explanatory_depth", 0))
    unsupported = _clamp_score("unsupported_claims_penalty", score.get("unsupported_claims_penalty", 0))
    verification = _clamp_score("verification_penalty", score.get("verification_penalty", 0))
    return correctness + mechanism + coherence + depth - unsupported - verification


def render_score_pills(score: dict) -> None:
    pills = [
        ("Correct", score_fmt(score, "correctness")),
        ("Mechanism", score_fmt(score, "mechanism_coverage")),
        ("Coherence", score_fmt(score, "logical_coherence")),
        ("Depth", score_fmt(score, "explanatory_depth")),
        ("Judge Penalty", score_fmt(score, "unsupported_claims_penalty")),
        ("Verifier Penalty", score_fmt(score, "verification_penalty")),
    ]
    html_pills = "".join(
        f'<span class="cc-score-pill">{html.escape(label)}: {html.escape(value)}</span>' for label, value in pills
    )
    hp(f'<div class="cc-banner-score-row">{html_pills}</div>')


def render_answer_card(
    model_name: str,
    answer: str,
    score: Optional[dict] = None,
    is_winner: bool = False,
    is_best_overall: bool = False,
    mode: str = "standard",
) -> None:
    card_cls = "cc-card"
    if is_best_overall:
        card_cls += " best"
    elif is_winner:
        card_cls += " winner"

    badges = ""
    if is_best_overall:
        badges += '<span class="cc-badge best-b">⭐ Best Answer</span>'
    elif is_winner:
        badges += '<span class="cc-badge win-b">✓ Winner</span>'

    total = compute_total_score(score) if score else None
    score_html = f'<span class="cc-inline-score">Score {total:.1f}</span>' if total is not None else ""

    answer_safe = html.escape(clean_answer(answer, mode=mode)) if answer else "<i>No response</i>"

    scores_html = ""
    if score:
        pills = [
            ("Correct", score_fmt(score, "correctness")),
            ("Mechanism", score_fmt(score, "mechanism_coverage")),
            ("Coherence", score_fmt(score, "logical_coherence")),
            ("Depth", score_fmt(score, "explanatory_depth")),
            ("Judge Penalty", score_fmt(score, "unsupported_claims_penalty")),
            ("Verifier Penalty", score_fmt(score, "verification_penalty")),
        ]
        pills_html = "".join(
            f'<span class="cc-score-pill">{html.escape(label)}: {html.escape(value)}</span>'
            for label, value in pills
        )
        scores_html = f'<div class="cc-banner-score-row">{pills_html}</div>'

    hp(
        f"""
        <div class="{card_cls}">
            <div class="cc-card-head">
                <div class="cc-card-model-wrap cc-result-line">
                    <span class="cc-model-name">{html.escape(model_name)}</span>
                    {score_html}
                </div>
                <div class="cc-badge-row">{badges}</div>
            </div>
            <div class="cc-card-body">{answer_safe}</div>
            {scores_html}
        </div>
        """
    )


def render_answer_viewer(
    model_name: str,
    answer: str,
    score: Optional[dict] = None,
    is_winner: bool = False,
    is_best_overall: bool = False,
    mode: str = "standard",
) -> None:
    bits = [model_name]
    if is_winner:
        bits.append("Winner")
    if is_best_overall:
        bits.append("Best")

    title = " · ".join(bits)

    with st.expander(title, expanded=False):
        render_answer_card(
            model_name=model_name,
            answer=answer,
            score=score,
            is_winner=is_winner,
            is_best_overall=is_best_overall,
            mode=mode,
        )


def render_judge_panel(title: str, judge: dict) -> None:
    winner = judge.get("winner", "n/a")
    qtype = judge.get("question_type", "unknown")
    reason = judge.get("winner_reason", "")

    type_badge = f'<span class="cc-badge score-b">type: {html.escape(str(qtype))}</span>'
    win_cls = "win-b" if winner != "n/a" else "score-b"
    win_badge = f'<span class="cc-badge {win_cls}">winner: {html.escape(str(winner))}</span>'

    hp(
        f"""
        <div class="cc-judge-card">
            <div class="cc-judge-title">{html.escape(title)}</div>
            <div class="cc-badge-row" style="margin-bottom:.6rem;">{type_badge} {win_badge}</div>
            <div class="cc-judge-reason">{html.escape(str(reason))}</div>
        </div>
        """
    )


def _build_label_map_from_answers(answers: Dict[str, str]) -> Dict[str, str]:
    return {MODEL_LABELS[i]: model_name for i, model_name in enumerate(answers.keys())}


def _map_label_mentions(text: str, label_map: Dict[str, str]) -> str:
    if not text:
        return text

    out = text
    for label, model_name in sorted(label_map.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(f"{label}'s", f"{model_name}'s")
        out = out.replace(f"{label}\u2019s", f"{model_name}'s")
        if len(label) == 1:
            out = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])",
                model_name,
                out,
            )
        else:
            out = re.sub(rf"\b{re.escape(label)}\b", model_name, out)
    return out


def normalize_judge_output(judge: dict, answers: Dict[str, str]) -> dict:
    """
    Convert judge labels like A/B/C into actual model names.
    """
    judge = dict(judge or {})
    scores = dict(judge.get("scores") or {})

    label_map = (
        judge.get("label_mapping")
        or judge.get("label_map")
        or judge.get("model_map")
        or judge.get("answer_labels")
        or {}
    )

    if not label_map:
        label_keys = set(scores.keys())
        if label_keys and all(isinstance(k, str) and len(k) == 1 and k.isalpha() for k in label_keys):
            label_map = _build_label_map_from_answers(answers)

    normalized_scores = {}
    for key, val in scores.items():
        actual_name = label_map.get(key, key)
        if isinstance(val, dict):
            val = dict(val)
            for text_key in ("justification", "reason", "explanation", "winner_reason"):
                if isinstance(val.get(text_key), str):
                    mapped = _map_label_mentions(val[text_key], label_map)
                    val[text_key] = re.sub(r"(?<![A-Za-z0-9_])[A-Z](?![A-Za-z0-9_])", "it", mapped)
        normalized_scores[actual_name] = val

    judge["scores"] = normalized_scores
    judge["label_mapping"] = label_map

    winner = judge.get("winner")
    if isinstance(winner, str):
        judge["winner"] = label_map.get(winner, winner)

    winner_reason = judge.get("winner_reason")
    if isinstance(winner_reason, str):
        mapped = _map_label_mentions(winner_reason, label_map)
        judge["winner_reason"] = re.sub(r"(?<![A-Za-z0-9_])[A-Z](?![A-Za-z0-9_])", "it", mapped)

    return judge

def find_best_overall(std: dict, cot: dict) -> Tuple[str, str, str, str, float]:
    best_model = ""
    best_mode = ""
    best_answer = ""
    best_reason = ""
    best_score = float("-inf")

    for mode, payload in [("standard", std), ("cot", cot)]:
        judge = payload.get("judge", {}) or {}
        answers = payload.get("answers", {}) or {}
        scores = judge.get("scores", {}) or {}
        label_map = judge.get("label_mapping") or {}

        for model_name, sc in scores.items():
            total = compute_total_score(sc)

            should_replace = total > best_score
            if total == best_score and mode == "cot" and best_mode != "cot":
                should_replace = True

            if should_replace:
                best_score = total
                best_model = model_name
                best_mode = mode
                best_answer = answers.get(model_name, "") or ""
                raw_reason = sc.get("justification", "") or judge.get("winner_reason", "") or ""
                mapped_reason = _map_label_mentions(raw_reason, label_map) if label_map else raw_reason
                best_reason = re.sub(r"(?<![A-Za-z0-9_])[A-Z](?![A-Za-z0-9_])", "it", mapped_reason)

    return best_model, best_mode, best_answer, best_reason, best_score


def render_best_banner(
    model: str,
    mode: str,
    answer: str,
    reason: str,
    score: float,
    score_dict: Optional[dict] = None,
) -> None:
    score_html = ""
    if score_dict:
        pills = [
            ("Correct", score_fmt(score_dict, "correctness")),
            ("Mechanism", score_fmt(score_dict, "mechanism_coverage")),
            ("Coherence", score_fmt(score_dict, "logical_coherence")),
            ("Depth", score_fmt(score_dict, "explanatory_depth")),
            ("Judge Penalty", score_fmt(score_dict, "unsupported_claims_penalty")),
            ("Verifier Penalty", score_fmt(score_dict, "verification_penalty")),
        ]
        score_html = "".join(
            f'<span class="cc-score-pill">{html.escape(label)}: {html.escape(value)}</span>'
            for label, value in pills
        )
        score_html = f'<div class="cc-banner-score-row">{score_html}</div>'

    reason_text = reason if reason else "Highest composite score across all models and modes."

    hp(
        f"""
        <div class="cc-best-banner">
            <div class="cc-best-icon">⭐</div>
            <div class="cc-best-content">
                <div class="cc-best-title">Best Answer Overall</div>
                <div class="cc-result-line">
                    <span class="cc-model-name">{html.escape(model)}</span>
                    <span style="font-size:.83rem;color:#fcd34d;font-weight:400;">· {html.escape(mode.title())}</span>
                    <span class="cc-inline-score">Score {score:.1f}</span>
                </div>
                <div class="cc-best-answer">{html.escape(answer)}</div>
                {score_html}
                <div class="cc-best-reason"><strong>Reason:</strong> {html.escape(reason_text)}</div>
            </div>
        </div>
        """
    )

def sync_question_state() -> None:
    st.session_state.has_question = bool(st.session_state.question_input.strip())

def render_mode_column(payload: dict, best_model: str, best_mode: str, mode: str) -> None:
    judge = payload.get("judge", {}) or {}
    winner = judge.get("winner")
    scores = judge.get("scores") or {}
    answers = payload.get("answers", {}) or {}

    other_models = [
        model_name
        for model_name in answers.keys()
        if not (model_name == best_model and mode == best_mode)
    ]

    if other_models:
        for model_name in other_models:
            render_answer_viewer(
                model_name=model_name,
                answer=answers.get(model_name, ""),
                score=scores.get(model_name),
                is_winner=(model_name == winner),
                is_best_overall=False,
                mode=mode,
            )


def render_both_view(std: dict, cot: dict, best_model: str, best_mode: str) -> None:
    left, mid, right = st.columns([1, 0.03, 1], vertical_alignment="top")

    with left:
        render_mode_column(std, best_model, best_mode, "standard")

    with mid:
        hp('<div class="cc-divider"></div>')

    with right:
        render_mode_column(cot, best_model, best_mode, "cot")


def render_sidebar_summary(profile: str, verifier_on: bool, generation_mode: str) -> None:
    verifier_label = "On" if verifier_on else "Off"
    mode_label = generation_mode.title()
    profile_note = {
        "Balanced": "Default review rhythm with a stable judge path.",
        "Strict": "A firmer evaluation path for closer comparisons.",
        "Exploratory": "A more varied judge path for stress-testing outputs.",
    }.get(profile, "Balanced review path.")

    hp(
        f"""
        <div class="cc-sidebar-card">
            <div class="cc-sidebar-card-title">Session profile</div>
            <div class="cc-sidebar-card-body">
                <div><strong>{html.escape(profile)}</strong> · {html.escape(profile_note)}</div>
                <div style="margin-top:.35rem;">Mode: <strong>{html.escape(mode_label)}</strong></div>
                <div>Verifier: <strong>{html.escape(verifier_label)}</strong></div>
            </div>
        </div>
        """
    )


def check_question(question: str) -> dict:
    judge_client = cached_judge_client()
    try:
        return run_gatekeeper(judge_client, question)
    except Exception as exc:
        return {
            "allowed": False,
            "reason": f"Gatekeeper failed: {exc}",
        }


def run_one(question: str, mode: str, use_verifier: bool, review_style: str = "Balanced") -> dict:
    cfg = cached_config()
    answer_clients = cached_answer_clients()
    judge_client = cached_judge_client()
    verifier_client = cached_verifier_client()

    answers: Dict[str, str] = {}
    answer_errors: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max(1, len(answer_clients))) as ex:
        futs = {ex.submit(generate_answer, c, question, mode=mode): n for n, c in answer_clients.items()}
        for f in as_completed(futs):
            name = futs[f]
            try:
                raw = f.result()
                answers[name] = clean_answer(raw, mode=mode)
            except Exception as exc:
                answer_errors[name] = str(exc)
                answers[name] = ""

    verifications: Dict[str, dict] = {}
    verification_errors: Dict[str, str] = {}
    if use_verifier and answers:
        with ThreadPoolExecutor(max_workers=min(3, max(1, len(answers)))) as ex:
            futs2 = {ex.submit(run_verifier, verifier_client, question, a): n for n, a in answers.items()}
            for f in as_completed(futs2):
                name = futs2[f]
                try:
                    verifications[name] = f.result()
                except Exception as exc:
                    verification_errors[name] = str(exc)

    try:
        judge, _ = run_judge(
            judge_client,
            question,
            answers,
            seed=cfg.seed,
            review_style=review_style,
        )
    except Exception as exc:
        judge = {
            "winner": "n/a",
            "question_type": "unknown",
            "winner_reason": f"Judge failed: {exc}",
            "scores": {},
        }

    if use_verifier and verifications:
        try:
            judge = attach_verifications(judge, verifications)
        except Exception:
            pass

    judge = normalize_judge_output(judge, answers)
    judge = resolve_tie_break(
    judge,
    question=question,
    use_verifier=bool(use_verifier and verifications),
    )
    if answer_errors:
        judge.setdefault("run_errors", {})
        judge["run_errors"]["answers"] = answer_errors
    if verification_errors:
        judge.setdefault("run_errors", {})
        judge["run_errors"]["verifications"] = verification_errors

    return {"answers": answers, "judge": judge, "verifications": verifications}


def render_input_page(question_bank) -> None:
    def load_sample_question() -> None:
        if question_bank:
            sample = random.choice(question_bank)
            st.session_state.question_input = sample.get("question", "")
            sync_question_state()

    def clear_question() -> None:
        st.session_state.chat_history = []
        st.session_state.question_input = ""
        st.session_state.latest_run = None
        st.session_state.page = "input"
        st.session_state.has_question = False
    section_heading("Submit a Question")
    hp('<div class="cc-question-helper">Enter one clear scientific question, claim, or mechanism to test across models.</div>')

    st.text_area(
    "q",
    height=170,
    placeholder="e.g. Why does increasing CO₂ accelerate plant growth only up to a point?",
    label_visibility="collapsed",
    key="question_input",
    on_change=sync_question_state,
    )

    has_question = st.session_state.get("has_question", False)

    _, c1, c2, c3, _ = st.columns([1.8, 1.45, 1.45, 1.45, 1.8], gap="small")

    with c1:
        st.button("Load sample", use_container_width=True, on_click=load_sample_question)

    with c2:
        run_clicked = st.button("Run the Court", use_container_width=True, disabled=not has_question)

    with c3:
        st.button("Clear", use_container_width=True, on_click=clear_question)


    if run_clicked and has_question:
        # q = question.strip()
        q = st.session_state.get("question_input", "").strip()
        status = st.empty()
        prog = st.empty()
        status.info("Checking whether the question is appropriate...")
        prog.progress(5)

        gatekeeper = check_question(q)

        if not gatekeeper.get("allowed", False):
            record = {
                "question": q,
                "generation_mode": st.session_state.generation_mode,
                "results": {},
                "blocked": True,
                "gatekeeper": gatekeeper,
                "default_reply": DEFAULT_REJECT_REPLY,
            }

            history = st.session_state.get("chat_history", [])
            history.append(record)
            st.session_state.chat_history = history
            st.session_state.latest_run = record
            st.session_state.page = "results"

            prog.progress(100)
            status.warning("Question blocked by gatekeeper.")
            st.rerun()

        selected_modes = modes_for_selection(st.session_state.generation_mode)

        status.info(f"Running {st.session_state.generation_mode.title()} mode...")
        prog.progress(20)

        results: Dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=len(selected_modes)) as ex:
            futs = {
                ex.submit(run_one, q, mode, st.session_state.use_verifier, st.session_state.court_profile): mode
                for mode in selected_modes
            }
            for f in as_completed(futs):
                mode = futs[f]
                try:
                    results[mode] = f.result()
                except Exception as exc:
                    results[mode] = {
                        "answers": {},
                        "judge": {
                            "winner": "n/a",
                            "question_type": "unknown",
                            "winner_reason": f"Run failed for {mode}: {exc}",
                            "scores": {},
                        },
                        "verifications": {},
                    }

        prog.progress(100)
        status.success("Court session complete.")

        history = st.session_state.get("chat_history", [])
        history.append(
            {
                "question": q,
                "generation_mode": st.session_state.generation_mode,
                "results": results,
            }
        )
        st.session_state.chat_history = history
        st.session_state.latest_run = history[-1]
        st.session_state.page = "results"
        st.rerun()


def render_results_page(latest: dict) -> None:
    if latest.get("blocked"):
        ctrl_left, ctrl_right = st.columns([0.82, 0.18], vertical_alignment="center")
        with ctrl_left:
            st.write("")
        with ctrl_right:
            if st.button("← Back to input", use_container_width=True, key="back_to_input_btn"):
                st.session_state.page = "input"
                st.rerun()

        section_heading("Results")
        render_question_card(latest.get("question", ""))

        st.warning("This question was blocked by the gatekeeper.")
        st.markdown(f"**Reason:** {html.escape(latest.get('gatekeeper', {}).get('reason', 'No reason provided.'))}")
        return

    results = latest["results"]
    generation_mode = latest["generation_mode"]
    ctrl_left, ctrl_right = st.columns([0.82, 0.18], vertical_alignment="center")
    with ctrl_left:
        st.write("")
    with ctrl_right:
        if st.button("← Back to input", use_container_width=True, key="back_to_input_btn"):
            st.session_state.page = "input"
            st.rerun()

    section_heading("Results")

    render_question_card(latest["question"])

    if generation_mode == "both":
        std = results.get("standard", {})
        cot = results.get("cot", {})

        best_model, best_mode, best_answer, best_reason, best_score = find_best_overall(std, cot)
        if best_model:
            best_payload = std if best_mode == "standard" else cot
            best_answer = best_payload.get("answers", {}).get(best_model, "")
            best_score_dict = best_payload.get("judge", {}).get("scores", {}).get(best_model, {})
            render_best_banner(best_model, best_mode, best_answer, best_reason, best_score, best_score_dict)
        section_heading("Other Model Answers")
        tab_std, tab_cot = st.tabs(["Standard", "CoT"])

        with tab_std:
            render_mode_column(std, best_model, best_mode, "standard")

        with tab_cot:
            render_mode_column(cot, best_model, best_mode, "cot")

        section_heading("Mode Comparison")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Standard Winner", std.get("judge", {}).get("winner", "n/a"))
        with m2:
            st.metric("CoT Winner", cot.get("judge", {}).get("winner", "n/a"))
        with m3:
            same = std.get("judge", {}).get("winner") == cot.get("judge", {}).get("winner")
            st.metric("Agreement", "✓ Yes" if same else "✗ No")
    else:
        only_mode = next(iter(results.keys()))
        payload = results[only_mode]

        best_model, best_mode, best_answer, best_reason, best_score = find_best_overall(
            payload if only_mode == "standard" else {},
            payload if only_mode == "cot" else {},
        )
        if best_model:
            best_answer = payload.get("answers", {}).get(best_model, "")
            best_score_dict = payload.get("judge", {}).get("scores", {}).get(best_model, {})
            render_best_banner(best_model, best_mode, best_answer, best_reason, best_score, best_score_dict)
        section_heading("Other Model Answers")
        render_mode_column(payload, best_model, best_mode, only_mode)

    section_heading("Raw Output")
    with st.expander("JSON output", expanded=False):
        for mode_name, payload in results.items():
            st.subheader(mode_name.title())
            st.json(payload)


def main() -> None:
    st.set_page_config(
        page_title="Cognitive Court",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()
    render_hero()

    for key, default in [
        ("chat_history", []),
        ("question_input", ""),
        ("page", "input"),
        ("latest_run", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default
    if "has_question" not in st.session_state:
        st.session_state.has_question = False

    if "generation_mode" not in st.session_state:
        st.session_state.generation_mode = "both"
    else:
        st.session_state.generation_mode = normalize_mode(st.session_state.generation_mode)

    if "use_verifier" not in st.session_state:
        st.session_state.use_verifier = True
    if "court_profile" not in st.session_state:
        st.session_state.court_profile = "Balanced"

    question_bank = cached_questions(DEFAULT_QUESTIONS_PATH)

    with st.sidebar:
        st.markdown("## Court Settings")
        st.caption("Choose generation mode, verifier and review style.")

        current_mode = normalize_mode(st.session_state.generation_mode)
        st.session_state.generation_mode = st.radio(
            "Generation mode",
            MODE_OPTIONS,
            index=MODE_OPTIONS.index(current_mode),
            format_func=lambda x: x.title(),
            help="Choose whether to generate Standard, CoT or Both.",
        )
        st.session_state.use_verifier = st.toggle("Claim verifier", value=st.session_state.use_verifier)
        profile_options = ["Balanced", "Strict", "Exploratory"]
        st.segmented_control(
            "Review style",
            options=profile_options,
            key="court_profile",
        )
        render_sidebar_summary(
            st.session_state.court_profile,
            st.session_state.use_verifier,
            st.session_state.generation_mode,
        )

    hp('<div class="cc-main">')

    if st.session_state.page == "results" and st.session_state.latest_run:
        render_results_page(st.session_state.latest_run)
    else:
        render_input_page(question_bank)
        if not st.session_state.get("chat_history"):
            st.info("Submit a question above to begin a court session.")

    hp("</div>")


if __name__ == "__main__":
    main()