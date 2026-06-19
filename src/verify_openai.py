"""Quick check that OpenAI gpt-4o-mini is configured and reachable."""

from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    if not config.model.api_key:
        print("OPENAI_API_KEY chưa được set trong .env")
        print("Mở file .env ở root repo và điền: OPENAI_API_KEY=sk-...")
        raise SystemExit(1)

    print(f"Provider: {config.model.provider}")
    print(f"Model:    {config.model.model_name}")
    print(f"Key:      {config.model.api_key[:8]}...{config.model.api_key[-4:]}")

    baseline = BaselineAgent(config=config, force_offline=False)
    advanced = AdvancedAgent(config=config, force_offline=False)

    print(f"Baseline live agent: {'OK' if baseline.langchain_agent else 'FAILED'}")
    print(f"Advanced live agent: {'OK' if advanced.langchain_agent else 'FAILED'}")

    if not advanced.langchain_agent:
        raise SystemExit(2)

    result = advanced.reply("verify-user", "verify-thread", "Chào, trả lời một câu ngắn: bạn đang dùng model gì?")
    print(f"Mode:   {result.get('mode')}")
    print(f"Answer: {result.get('answer', '')[:300]}")


if __name__ == "__main__":
    main()
