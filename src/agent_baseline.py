from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_runtime import (
    build_baseline_langgraph_agent,
    can_build_live_agent,
    invoke_agent,
    last_assistant_message,
)
from config import LabConfig, load_config
from memory_store import estimate_tokens


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-session memory only, no persistent User.md."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def _estimate_session_prompt_tokens(self, session: SessionState) -> int:
        return sum(estimate_tokens(item.get("content", "")) for item in session.messages)

    def _offline_response(self, session: SessionState, message: str) -> str:
        lowered = message.lower()

        if any(key in lowered for key in ("tên gì", "tên mình", "tên của mình")):
            for item in session.messages:
                if item["role"] != "user":
                    continue
                match_text = item["content"]
                if "tên" in match_text.lower():
                    return f"Trong phiên này bạn có nhắc: {match_text[:120]}"
            return "Trong phiên này mình chưa thấy bạn nói tên."

        if any(key in lowered for key in ("ở đâu", "nơi ở")):
            for item in reversed(session.messages):
                if item["role"] != "user":
                    continue
                if "ở" in item["content"].lower():
                    return f"Theo phiên hiện tại: {item['content'][:120]}"
            return "Trong phiên này chưa có thông tin nơi ở."

        if "style" in lowered or "trả lời" in lowered:
            for item in reversed(session.messages):
                if item["role"] != "user":
                    continue
                if "trả lời" in item["content"].lower() or "style" in item["content"].lower():
                    return f"Style bạn nhắc trong phiên: {item['content'][:120]}"
            return "Trong phiên này chưa có preference rõ về style."

        return "Đã ghi nhận trong phiên hiện tại. Baseline chỉ nhớ trong cùng thread."

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})

        prompt_tokens = self._estimate_session_prompt_tokens(session)
        session.prompt_tokens_processed += prompt_tokens

        answer = self._offline_response(session, message)
        session.messages.append({"role": "assistant", "content": answer})

        answer_tokens = estimate_tokens(answer)
        session.token_usage += answer_tokens

        return {
            "answer": answer,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "mode": "offline",
        }

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})

        result = invoke_agent(self.langchain_agent, message, thread_id)
        answer = last_assistant_message(result) or "Không nhận được phản hồi từ agent."
        session.messages.append({"role": "assistant", "content": answer})

        prompt_tokens = sum(estimate_tokens(item.get("content", "")) for item in session.messages)
        session.prompt_tokens_processed += prompt_tokens

        answer_tokens = estimate_tokens(answer)
        session.token_usage += answer_tokens
        return {
            "answer": answer,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "mode": "live",
        }

    def _maybe_build_langchain_agent(self):
        if not can_build_live_agent(self.config):
            return None
        try:
            return build_baseline_langgraph_agent(self.config)
        except Exception:
            return None
