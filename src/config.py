from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    profile_confidence_threshold: float
    memory_decay_turns: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _provider_from_env(prefix: str = "LLM") -> ProviderConfig:
    provider = normalize_provider(os.getenv(f"{prefix}_PROVIDER", "openai"))
    model_name = os.getenv(f"{prefix}_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv(f"{prefix}_TEMPERATURE", "0.2"))

    api_key: str | None = None
    base_url: str | None = None

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "custom":
        api_key = os.getenv("CUSTOM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("CUSTOM_BASE_URL")
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    load_dotenv(root / ".env")

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "6"))
    profile_confidence_threshold = float(os.getenv("PROFILE_CONFIDENCE_THRESHOLD", "0.65"))
    memory_decay_turns = int(os.getenv("MEMORY_DECAY_TURNS", "25"))

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        profile_confidence_threshold=profile_confidence_threshold,
        memory_decay_turns=memory_decay_turns,
        model=_provider_from_env("LLM"),
        judge_model=_provider_from_env("JUDGE"),
    )
