from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from config import LabConfig
from memory_store import UserProfileStore, estimate_tokens
from model_provider import build_chat_model

BASELINE_SYSTEM_PROMPT = """You are Baseline Agent for a memory lab.

Rules:
- Remember only the current conversation thread (short-term memory).
- You do NOT have User.md or any cross-session persistent memory.
- If asked about facts from a previous session/thread, say you cannot remember them.
- Reply concisely in Vietnamese when the user writes Vietnamese.
"""

ADVANCED_SYSTEM_PROMPT = """You are Advanced Agent for a memory lab.

You have three memory layers:
1. Short-term thread memory (conversation in this thread).
2. Persistent User.md profile (use read_user_profile / update_user_fact tools).
3. Compact/summarized history for long threads.

Rules:
- Store stable facts (name, location, profession, preferences) via update_user_fact.
- Do not store question-only turns as facts.
- Prefer corrections over older conflicting facts.
- Reply concisely in Vietnamese when the user writes Vietnamese.
- When recalling profile facts, use User.md first.
"""


@dataclass
class ProfileToolContext:
    user_id: str = "default"


def build_profile_tools(profile_store: UserProfileStore, context: ProfileToolContext):
    @tool
    def read_user_profile() -> str:
        """Read the persistent User.md profile for the current user."""

        return profile_store.read_text(context.user_id)

    @tool
    def update_user_fact(key: str, value: str) -> str:
        """Upsert one stable fact into User.md (e.g. name, location, profession)."""

        profile_store.upsert_fact(context.user_id, key, value)
        return f"Updated {key} in User.md."

    @tool
    def edit_user_profile(search_text: str, replacement: str) -> str:
        """Replace one occurrence of text inside User.md."""

        changed = profile_store.edit_text(context.user_id, search_text, replacement)
        return "Profile updated." if changed else "Search text not found in User.md."

    return [read_user_profile, update_user_fact, edit_user_profile]


def _resolve_model(config: LabConfig, model: BaseChatModel | None) -> BaseChatModel:
    if model is not None:
        return model
    return build_chat_model(config.model)


def can_build_live_agent(config: LabConfig) -> bool:
    if config.model.provider == "ollama":
        return True
    return bool(config.model.api_key)


def build_baseline_langgraph_agent(
    config: LabConfig,
    *,
    model: BaseChatModel | None = None,
):
    chat_model = _resolve_model(config, model)
    return create_agent(
        chat_model,
        tools=[],
        system_prompt=BASELINE_SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


def build_advanced_langgraph_agent(
    config: LabConfig,
    profile_store: UserProfileStore,
    tool_context: ProfileToolContext,
    *,
    model: BaseChatModel | None = None,
):
    chat_model = _resolve_model(config, model)
    tools = build_profile_tools(profile_store, tool_context)
    middleware = [
        SummarizationMiddleware(
            chat_model,
            trigger=("tokens", config.compact_threshold_tokens),
            keep=("messages", config.compact_keep_messages),
        )
    ]
    return create_agent(
        chat_model,
        tools=tools,
        system_prompt=ADVANCED_SYSTEM_PROMPT,
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )


def invoke_agent(agent, message: str, thread_id: str) -> dict[str, Any]:
    return agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": thread_id}},
    )


def last_assistant_message(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    for item in reversed(messages):
        if isinstance(item, AIMessage):
            content = item.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [block.get("text", "") for block in content if isinstance(block, dict)]
                return "".join(parts).strip()
            return str(content)
        if isinstance(item, BaseMessage) and item.type == "ai":
            return str(item.content)
    return ""


def estimate_result_prompt_tokens(result: dict[str, Any]) -> int:
    messages = result.get("messages", [])
    total = 0
    for item in messages:
        if isinstance(item, BaseMessage):
            total += estimate_tokens(str(item.content))
    return total
