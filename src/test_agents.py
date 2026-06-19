from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, apply_profile_candidates, extract_profile_candidates


def make_config(tmp_path: Path) -> LabConfig:
    base = load_config(Path(__file__).resolve().parent.parent)
    return replace(
        base,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=120,
        compact_keep_messages=2,
        profile_confidence_threshold=0.65,
        memory_decay_turns=3,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / "profiles")

    assert store.read_text("alice") == "# User Profile\n\n"

    store.write_text("alice", "# User Profile\n\n- name: Alice\n")
    assert "Alice" in store.read_text("alice")
    assert store.file_size("alice") > 0

    store.upsert_fact("alice", "location", "Huế")
    assert store.facts("alice")["location"] == "Huế"

    changed = store.edit_text("alice", "Huế", "Đà Nẵng")
    assert changed is True
    assert store.facts("alice")["location"] == "Đà Nẵng"


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = CompactMemoryManager(
        threshold_tokens=config.compact_threshold_tokens,
        keep_messages=config.compact_keep_messages,
    )

    thread_id = "long-thread"
    long_message = "Mình đang test compact memory với một câu rất dài. " * 20
    for index in range(8):
        manager.append(thread_id, "user", f"{long_message} turn {index}")

    ctx = manager.context(thread_id)
    messages = ctx["messages"]
    assert isinstance(messages, list)
    assert len(messages) <= config.compact_keep_messages
    assert manager.compaction_count(thread_id) >= 1
    assert str(ctx.get("summary", "")).strip() != ""


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)
    user_id = "dungct"

    baseline.reply(user_id, "thread-a", "Chào bạn, mình tên là DũngCT.")
    advanced.reply(user_id, "thread-a", "Chào bạn, mình tên là DũngCT.")

    baseline_answer = baseline.reply(user_id, "thread-b", "Mình tên gì?")["answer"]
    advanced_answer = advanced.reply(user_id, "thread-b", "Mình tên gì?")["answer"]

    assert "dũngct" not in baseline_answer.lower()
    assert "dũngct" in advanced_answer.lower()


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)
    user_id = "stress-user"
    thread_id = "stress-thread"
    long_message = "Tin tức dài về benchmark memory và trade-off token. " * 25

    for index in range(10):
        baseline.reply(user_id, thread_id, f"{long_message} #{index}")
        advanced.reply(user_id, thread_id, f"{long_message} #{index}")

    baseline_prompt = baseline.prompt_token_usage(thread_id)
    advanced_prompt = advanced.prompt_token_usage(thread_id)

    assert advanced.compaction_count(thread_id) >= 1
    assert advanced_prompt < baseline_prompt


def test_confidence_threshold_blocks_weak_inference(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / "profiles")
    user_id = "alice"

    candidates = extract_profile_candidates("Sáng nay mình uống cà phê rồi đi làm.")
    written = apply_profile_candidates(
        store,
        user_id,
        candidates,
        current_turn=1,
        threshold=config.profile_confidence_threshold,
    )

    assert "favorite_drink" not in written
    assert "favorite_drink" not in store.facts(user_id)


def test_correction_overrides_old_fact(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    user_id = "dungct"

    agent.reply(user_id, "t1", "Mình ở Đà Nẵng và đang làm backend engineer.")
    agent.reply(user_id, "t1", "Mình đính chính: giờ mình đang ở Huế.")
    agent.reply(user_id, "t1", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")

    facts = agent.profile_store.facts(user_id)
    assert facts["location"] == "Huế"
    assert facts["profession"] == "MLOps engineer"
    assert "backend engineer" not in facts["profession"]


def test_noise_location_is_ignored(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    user_id = "noise-user"

    agent.reply(user_id, "t1", "Mình tên là Test User.")
    agent.reply(
        user_id,
        "t1",
        "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày với đối tác chứ không phải nơi ở hiện tại.",
    )

    assert "hà nội" not in agent.profile_store.facts(user_id).get("location", "").lower()


def test_memory_decay_hides_stale_secondary_fact(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), memory_decay_turns=2)
    agent = AdvancedAgent(config=config, force_offline=True)
    user_id = "decay-user"

    agent.reply(user_id, "t1", "Mình quan tâm nhiều đến Python, AI agent và benchmark memory.")
    agent.reply(user_id, "t1", "Turn 2 filler message không thêm fact mới.")
    agent.reply(user_id, "t1", "Turn 3 filler message không thêm fact mới.")
    agent.reply(user_id, "t1", "Turn 4 filler message không thêm fact mới.")

    active = agent._active_facts(user_id)
    assert "interests" not in active
    assert "interests" in agent.profile_store.facts(user_id)


def test_langgraph_agents_build_and_invoke(tmp_path: Path) -> None:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from agent_runtime import (
        ProfileToolContext,
        build_advanced_langgraph_agent,
        build_baseline_langgraph_agent,
        invoke_agent,
        last_assistant_message,
    )

    class ToolCapableFakeChatModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    config = make_config(tmp_path)
    responses = [AIMessage(content="Xin chao, minh la agent live.")] * 5
    fake_model = ToolCapableFakeChatModel(messages=iter(responses))

    baseline = build_baseline_langgraph_agent(config, model=fake_model)
    baseline_result = invoke_agent(baseline, "Chào bạn", "live-thread")
    assert "agent live" in last_assistant_message(baseline_result).lower()

    store = UserProfileStore(config.state_dir / "profiles")
    ctx = ProfileToolContext(user_id="live-user")
    advanced = build_advanced_langgraph_agent(config, store, ctx, model=fake_model)
    advanced_result = invoke_agent(advanced, "Chào bạn", "live-advanced")
    assert last_assistant_message(advanced_result)
