from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    normalized_answer = answer.lower()
    hits = sum(1 for item in expected if item.lower() in normalized_answer)
    if hits == 0:
        return 0.0
    if hits == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    recall = recall_points(answer, expected)
    if not answer.strip():
        return 0.0

    score = recall
    lowered = answer.lower()
    if any(marker in lowered for marker in ("- ", "bullet", "trade-off", "ngắn")):
        score += 0.15
    if 20 <= len(answer) <= 500:
        score += 0.1
    return min(1.0, score)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    memory_growth = 0
    compactions = 0

    for conversation in conversations:
        user_id = conversation["user_id"]
        thread_id = conversation["id"]
        initial_memory = agent.memory_file_size(user_id) if hasattr(agent, "memory_file_size") else 0

        for turn in conversation.get("turns", []):
            result = agent.reply(user_id, thread_id, turn)
            total_agent_tokens += int(result.get("agent_tokens", 0))

        if hasattr(agent, "memory_file_size"):
            memory_growth = max(memory_growth, agent.memory_file_size(user_id) - initial_memory)

        compactions += agent.compaction_count(thread_id)
        total_prompt_tokens += agent.prompt_token_usage(thread_id)

        recall_thread = f"{thread_id}-recall"
        for item in conversation.get("recall_questions", []):
            question = item["question"]
            expected = item.get("expected_contains", [])
            answer = agent.reply(user_id, recall_thread, question)["answer"]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=compactions,
    )


def format_rows(title: str, rows: list[BenchmarkRow]) -> str:
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_rows = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]

    if tabulate:
        return f"### {title}\n\n" + tabulate(table_rows, headers=headers, tablefmt="github")
    lines = [f"### {title}", "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in table_rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    standard_conversations = load_conversations(standard_path)
    stress_conversations = load_conversations(stress_path)

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    standard_rows = [
        run_agent_benchmark("Baseline", baseline, standard_conversations, config),
        run_agent_benchmark("Advanced", advanced, standard_conversations, config),
    ]
    stress_rows = [
        run_agent_benchmark("Baseline", BaselineAgent(config=config, force_offline=True), stress_conversations, config),
        run_agent_benchmark("Advanced", AdvancedAgent(config=config, force_offline=True), stress_conversations, config),
    ]

    print("## Standard Benchmark\n")
    print(format_rows("Standard Benchmark", standard_rows))
    print("\n## Long-Context Stress Benchmark\n")
    print(format_rows("Long-Context Stress Benchmark", stress_rows))


if __name__ == "__main__":
    main()
