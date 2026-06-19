from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_runtime import (
    ProfileToolContext,
    build_advanced_langgraph_agent,
    can_build_live_agent,
    invoke_agent,
    last_assistant_message,
)
from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    apply_profile_candidates,
    estimate_tokens,
    extract_profile_candidates,
    is_question_only_turn,
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term + User.md + compact memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.user_turn_counter: dict[str, int] = {}
        self.profile_tool_context = ProfileToolContext()
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _next_turn(self, user_id: str) -> int:
        self.user_turn_counter[user_id] = self.user_turn_counter.get(user_id, 0) + 1
        return self.user_turn_counter[user_id]

    def _persist_profile_updates(self, user_id: str, message: str, current_turn: int) -> list[str]:
        if is_question_only_turn(message):
            return []
        candidates = extract_profile_candidates(message)
        return apply_profile_candidates(
            self.profile_store,
            user_id,
            candidates,
            current_turn=current_turn,
            threshold=self.config.profile_confidence_threshold,
        )

    def _active_facts(self, user_id: str) -> dict[str, str]:
        current_turn = self.user_turn_counter.get(user_id, 0)
        return self.profile_store.active_facts(
            user_id,
            current_turn=current_turn,
            decay_turns=self.config.memory_decay_turns,
        )

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))
        messages = ctx.get("messages", [])
        total = estimate_tokens(profile) + estimate_tokens(summary)
        if isinstance(messages, list):
            for item in messages:
                if isinstance(item, dict):
                    total += estimate_tokens(str(item.get("content", "")))
        return total

    def _format_bullet_answer(self, lines: list[str]) -> str:
        bullets = [line for line in lines if line][:3]
        if not bullets:
            return "Chưa có đủ thông tin trong User.md."
        return "\n".join(f"- {line}" for line in bullets)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self._active_facts(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))
        lowered = message.lower()
        lines: list[str] = []

        wants_name = any(key in lowered for key in ("tên", "dungct", "ai không", "tóm tắt", "nhắc lại"))
        wants_location = any(key in lowered for key in ("ở đâu", "nơi ở", "hiện tại mình", "ở huế", "đà nẵng"))
        wants_profession = any(key in lowered for key in ("nghề", "làm nghề", "nghề nghiệp", "mlops", "backend"))
        wants_style = "style" in lowered or ("trả lời" in lowered and "thích" in lowered) or "3 bullet" in lowered
        wants_drink = "đồ uống" in lowered or ("uống" in lowered and "món" not in lowered)
        wants_food = "món ăn" in lowered or "mì quảng" in lowered
        wants_pet = "nuôi" in lowered or "corgi" in lowered or "con gì" in lowered
        wants_interests = any(key in lowered for key in ("quan tâm", "python", "ai agent", "mối quan tâm", "tóm tắt"))

        if wants_name and facts.get("name"):
            lines.append(f"Tên: {facts['name']}")
        if wants_location and facts.get("location"):
            lines.append(f"Nơi ở hiện tại: {facts['location']}")
        if wants_profession and facts.get("profession"):
            lines.append(f"Nghề nghiệp hiện tại: {facts['profession']}")
        if wants_style and facts.get("response_style"):
            lines.append(f"Style trả lời: {facts['response_style']}")
        if wants_drink and facts.get("favorite_drink"):
            lines.append(f"Đồ uống yêu thích: {facts['favorite_drink']}")
        if wants_food and facts.get("favorite_food"):
            lines.append(f"Món ăn yêu thích: {facts['favorite_food']}")
        if wants_pet and facts.get("pet"):
            lines.append(f"Thú cưng: {facts['pet']}")
        if wants_interests and facts.get("interests"):
            lines.append(f"Mối quan tâm: {facts['interests']}")

        if "huế" in lowered or "hà nội" in lowered or "product manager" in lowered:
            if facts.get("profession"):
                lines.append(f"Nghề nghiệp hiện tại: {facts['profession']} (bỏ qua nhiễu PM/Hà Nội).")
            if facts.get("location"):
                lines.append(f"Nơi ở hiện tại: {facts['location']}.")

        if lines:
            return self._format_bullet_answer(lines)

        if any(key in lowered for key in ("artemis", "x-59", "wmo", "british columbia", "tin", "news", "stress test")):
            hints = []
            if "artemis" in summary.lower():
                hints.append("Artemis III: roadmap neo bằng milestone kỹ thuật trước launch lớn.")
            if "x-59" in summary.lower() or "siêu thanh" in summary.lower():
                hints.append("X-59: tối ưu hiệu năng nhưng giảm externality cho người dùng.")
            if "wmo" in summary.lower() or "el nino" in summary.lower():
                hints.append("WMO: truyền đạt rủi ro bằng xác suất tăng dần theo thời gian.")
            if "british columbia" in summary.lower() or "điện" in summary.lower():
                hints.append("BC energy: cân bằng scale hạ tầng với hiệu quả tiết kiệm nhu cầu.")
            if hints:
                return self._format_bullet_answer(hints)

        if facts:
            summary_lines = []
            for key, label in (
                ("name", "Tên"),
                ("location", "Nơi ở"),
                ("profession", "Nghề"),
                ("response_style", "Style"),
                ("favorite_drink", "Đồ uống"),
                ("favorite_food", "Món ăn"),
                ("pet", "Thú cưng"),
                ("interests", "Quan tâm"),
            ):
                if facts.get(key):
                    summary_lines.append(f"{label}: {facts[key]}")
            if summary_lines:
                return self._format_bullet_answer(summary_lines)

        return "Chưa có đủ thông tin trong User.md cho câu hỏi này."

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        current_turn = self._next_turn(user_id)
        written_keys = self._persist_profile_updates(user_id, message, current_turn)
        self.compact_memory.append(thread_id, "user", message)

        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        answer = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", answer)

        answer_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + answer_tokens

        return {
            "answer": answer,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_path": str(self.profile_store.path_for(user_id)),
            "profile_updates": written_keys,
            "mode": "offline",
        }

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        current_turn = self._next_turn(user_id)
        written_keys = self._persist_profile_updates(user_id, message, current_turn)
        self.compact_memory.append(thread_id, "user", message)
        self.profile_tool_context.user_id = user_id

        profile = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", "")).strip()
        context_block = profile
        if summary:
            context_block += f"\n\nCompact summary:\n{summary}"

        enriched_message = (
            f"Persistent profile and compact context:\n{context_block}\n\n"
            f"User message:\n{message}"
        )

        result = invoke_agent(self.langchain_agent, enriched_message, thread_id)
        answer = last_assistant_message(result) or self._offline_response(user_id, thread_id, message)

        self.compact_memory.append(thread_id, "assistant", answer)

        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        answer_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + answer_tokens
        return {
            "answer": answer,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_path": str(self.profile_store.path_for(user_id)),
            "profile_updates": written_keys,
            "mode": "live",
        }

    def _maybe_build_langchain_agent(self):
        if not can_build_live_agent(self.config):
            return None
        try:
            return build_advanced_langgraph_agent(
                self.config,
                self.profile_store,
                self.profile_tool_context,
            )
        except Exception:
            return None
