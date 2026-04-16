from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    base_url: str | None
    api_key: str | None
    model_id: str


@dataclass(frozen=True)
class AppConfig:
    qwen: ModelConfig
    mistral: ModelConfig
    llama: ModelConfig
    judge: ModelConfig
    temperature: float = 0.2
    answer_max_new_tokens: int = 500
    judge_max_new_tokens: int = 900
    verifier_max_new_tokens: int = 250
    seed: int = 42
    mock_mode: bool = False


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Missing environment variable: {name}")
    value = value.strip()
    if not value:
        raise ValueError(f"Empty environment variable: {name}")
    return value


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def load_config() -> AppConfig:
    mock_mode = os.getenv("MOCK_MODE", "0") == "1"

    # ---------------- Answer models → OpenRouter ----------------
    answer_base_url = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    answer_api_key = _env("OPENROUTER_API_KEY")

    # ---------------- Judge + Verifier → OpenAI ----------------
    judge_base_url = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    judge_api_key = _env("OPENAI_API_KEY")

    return AppConfig(
        qwen=ModelConfig(
            name="Qwen2.5-7B-Instruct",
            base_url=answer_base_url,
            api_key=answer_api_key,
            model_id=_env("QWEN_MODEL", "qwen/qwen-2.5-7b-instruct"),
        ),
        mistral=ModelConfig(
            name="Ministral-8B-2512",
            base_url=answer_base_url,
            api_key=answer_api_key,
            model_id=_env("MISTRAL_MODEL", "mistralai/ministral-8b-2512"),
        ),
        llama=ModelConfig(
            name="Llama-3.1-8B-Instruct",
            base_url=answer_base_url,
            api_key=answer_api_key,
            model_id=_env("LLAMA_MODEL", "meta-llama/llama-3.1-8b-instruct"),
        ),
        judge=ModelConfig(
            name="GPT-4o-mini",
            base_url=judge_base_url,
            api_key=judge_api_key,
            model_id=_env("JUDGE_MODEL", "gpt-4o-mini"),
        ),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        answer_max_new_tokens=int(os.getenv("ANSWER_MAX_NEW_TOKENS", "500")),
        judge_max_new_tokens=int(os.getenv("JUDGE_MAX_NEW_TOKENS", "900")),
        verifier_max_new_tokens=int(os.getenv("VERIFIER_MAX_NEW_TOKENS", "250")),
        seed=int(os.getenv("SEED", "42")),
        mock_mode=mock_mode,
    )