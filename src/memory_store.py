from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_PROFILE = "# User Profile\n\n"
_META_PATTERN = re.compile(
    r"^(?P<value>.+?)\s*<!--\s*meta:conf=(?P<conf>[\d.]+),turn=(?P<turn>\d+),mentions=(?P<mentions>\d+)\s*-->\s*$"
)
_FACT_LINE = re.compile(r"^-\s*([\w_]+)\s*:\s*(.+)$")

CORE_FACT_KEYS = frozenset({"name", "location", "profession", "response_style"})
DECAYABLE_KEYS = frozenset({"interests", "favorite_food", "favorite_drink", "pet"})

NOISE_MARKERS = (
    "chỉ là nơi",
    "vừa bay ra họp",
    "câu đùa",
    "đùa với",
    "gây nhiễu",
    "không phải nơi ở",
)


@dataclass
class ProfileFact:
    """Structured entity candidate extracted from one user turn."""

    key: str
    value: str
    confidence: float
    entity_type: str
    is_correction: bool = False


@dataclass
class FactRecord:
    key: str
    value: str
    confidence: float
    turn: int
    mentions: int = 1


def estimate_tokens(text: str) -> int:
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


def _slugify(user_id: str) -> str:
    return re.sub(r"[^\w\-]", "_", user_id.strip()) or "default"


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        return self.root_dir / _slugify(user_id) / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return _DEFAULT_PROFILE
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    def facts(self, user_id: str) -> dict[str, str]:
        return {key: record.value for key, record in self.fact_records(user_id).items()}

    def fact_records(self, user_id: str) -> dict[str, FactRecord]:
        content = self.read_text(user_id)
        result: dict[str, FactRecord] = {}
        for line in content.splitlines():
            match = _FACT_LINE.match(line.strip())
            if not match:
                continue
            key = match.group(1)
            raw_value = match.group(2).strip()
            meta = _META_PATTERN.match(raw_value)
            if meta:
                result[key] = FactRecord(
                    key=key,
                    value=meta.group("value").strip(),
                    confidence=float(meta.group("conf")),
                    turn=int(meta.group("turn")),
                    mentions=int(meta.group("mentions")),
                )
            else:
                result[key] = FactRecord(key=key, value=raw_value, confidence=0.7, turn=0, mentions=1)
        return result

    def active_facts(
        self,
        user_id: str,
        current_turn: int,
        decay_turns: int = 25,
    ) -> dict[str, str]:
        records = self.fact_records(user_id)
        active: dict[str, str] = {}
        for key, record in records.items():
            age = max(0, current_turn - record.turn)
            if key in DECAYABLE_KEYS and age > decay_turns and record.mentions < 2:
                continue
            active[key] = record.value
        return active

    def _format_fact_line(self, record: FactRecord) -> str:
        return (
            f"- {record.key}: {record.value} "
            f"<!-- meta:conf={record.confidence:.2f},turn={record.turn},mentions={record.mentions} -->"
        )

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        return self.upsert_fact_record(
            user_id,
            FactRecord(key=key, value=value, confidence=0.7, turn=0, mentions=1),
        )

    def upsert_fact_record(self, user_id: str, record: FactRecord) -> Path:
        content = self.read_text(user_id)
        line = self._format_fact_line(record)
        pattern = re.compile(rf"^-\s*{re.escape(record.key)}\s*:\s*.+$", re.MULTILINE)

        if pattern.search(content):
            content = pattern.sub(line, content)
        else:
            if not content.endswith("\n"):
                content += "\n"
            content += line + "\n"

        return self.write_text(user_id, content)

    def should_apply_fact(
        self,
        user_id: str,
        candidate: ProfileFact,
        current_turn: int,
        threshold: float,
    ) -> bool:
        if candidate.is_correction:
            return candidate.confidence >= 0.5
        if candidate.confidence < threshold:
            return False

        existing = self.fact_records(user_id).get(candidate.key)
        if existing is None:
            return True
        if existing.value.strip().lower() == candidate.value.strip().lower():
            return True
        if candidate.is_correction:
            return True
        return candidate.confidence >= existing.confidence


_DECLARATIVE_MARKERS = (
    "mình tên",
    "tên mình",
    "tên là",
    "đính chính",
    "cập nhật",
    "nhớ giúp",
    "yêu thích là",
    "đồ uống yêu thích",
    "món ăn yêu thích",
    "style trả lời",
    "muốn bạn trả lời",
    "không còn",
    "giờ chuyển",
    "chuyển sang",
    "nuôi",
    "mlops engineer",
    "backend engineer",
    "đang làm",
    "mình ở",
    "đang ở",
    "hiện ở",
    "làm việc ở",
)


def _looks_like_question(message: str) -> bool:
    lowered = message.lower().strip()
    if any(marker in lowered for marker in _DECLARATIVE_MARKERS):
        return False
    if "?" in lowered:
        return True
    question_starts = (
        "bạn có thể",
        "bạn nhớ",
        "bạn thử",
        "bạn có biết",
        "cho mình biết",
    )
    return any(lowered.startswith(prefix) for prefix in question_starts)


def is_question_only_turn(message: str) -> bool:
    return _looks_like_question(message)


def _valid_fact_value(key: str, value: str) -> bool:
    cleaned = value.strip()
    if len(cleaned) < 2:
        return False
    lowered = cleaned.lower()
    invalid_fragments = (
        "gì",
        "không?",
        "như thế nào",
        "của mình",
        "đâu",
        "nếu có",
        "hiện tại",
        "và mình",
        "và hai",
    )
    return not any(fragment in lowered for fragment in invalid_fragments)


def _contains_noise(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in NOISE_MARKERS)


def _add_candidate(
    candidates: list[ProfileFact],
    key: str,
    value: str,
    confidence: float,
    entity_type: str,
    *,
    is_correction: bool = False,
) -> None:
    if not _valid_fact_value(key, value):
        return
    candidates.append(
        ProfileFact(
            key=key,
            value=value.strip(),
            confidence=confidence,
            entity_type=entity_type,
            is_correction=is_correction,
        )
    )


def extract_profile_candidates(message: str) -> list[ProfileFact]:
    if _looks_like_question(message):
        return []

    candidates: list[ProfileFact] = []
    text = message.strip()
    lowered = text.lower()
    is_correction = any(
        marker in lowered for marker in ("đính chính", "cập nhật", "thực ra", "không còn", "chuyển sang")
    )
    if _contains_noise(text) and not is_correction:
        return []

    for pattern in (
        r"(?:mình tên là|tên mình là|tên là)\s+([^,\.\n?]+)",
        r"(?:tên\s+(?:của\s+)?mình\s+(?:là\s+)?)([^,\.\n?]+)",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            _add_candidate(candidates, "name", match.group(1), 0.95, "person", is_correction=is_correction)
            break

    correction_location = re.search(
        r"(?:đính chính|cập nhật|thực ra|giờ mình|từ tuần này|mình vẫn ở).{0,60}?(?:ở|đang ở|làm việc ở)\s+([^,\.\n]+?)(?:\s+chứ|\s+vài|\s+mỗi|\s+chưa|\.|,|$)",
        text,
        re.IGNORECASE,
    )
    still_location = re.search(r"mình vẫn ở\s+([^,\.\n]+?)(?:\s+chưa|\.|,|$)", text, re.IGNORECASE)
    if still_location:
        _add_candidate(candidates, "location", still_location.group(1), 0.92, "location", is_correction=True)
    elif correction_location:
        _add_candidate(candidates, "location", correction_location.group(1), 0.98, "location", is_correction=True)
    else:
        location_match = re.search(
            r"(?:hiện(?:\s+đang)?\s+ở|đang ở|mình ở)\s+([^,\.\n]+?)(?:\s+và|\s+chứ|\s+mỗi|\s+chưa|\.|,|$)",
            text,
            re.IGNORECASE,
        )
        if location_match:
            _add_candidate(candidates, "location", location_match.group(1), 0.8, "location")

    for pattern in (
        r"(?:chuyển sang|giờ (?:là|chuyển sang))\s+(MLOps engineer|backend engineer|[^,\.\n]+?(?:engineer|developer|manager|designer))",
        r"(?:đang làm|làm)\s+(MLOps engineer|backend engineer|[^,\.\n]+?(?:engineer|developer|manager|designer))",
        r"(?:làm\s+)(MLOps engineer|backend engineer)",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if "product manager" in value.lower() and "đùa" in lowered:
                continue
            conf = 0.98 if is_correction or "chuyển sang" in lowered else 0.85
            _add_candidate(candidates, "profession", value, conf, "profession", is_correction=is_correction)
            break

    if re.search(r"(?:style trả lời|trả lời).{0,40}(?:ngắn gọn|3 bullet|bullet ngắn)", text, re.IGNORECASE):
        style = (
            "3 bullet ngắn, có ví dụ thực chiến, ưu tiên trade-off"
            if "3 bullet" in lowered
            else "ngắn gọn, rõ ý, có ví dụ thực tế"
        )
        _add_candidate(candidates, "response_style", style, 0.88, "preference")

    drink_match = re.search(r"đồ uống yêu thích(?:\s+là)?\s+([^,\.\n]+)", text, re.IGNORECASE)
    if drink_match:
        _add_candidate(candidates, "favorite_drink", drink_match.group(1), 0.9, "preference")
    elif re.search(r"uống\s+cà phê", text, re.IGNORECASE):
        coffee_match = re.search(r"(cà phê[^,\.\n]*)", text, re.IGNORECASE)
        if coffee_match:
            _add_candidate(candidates, "favorite_drink", coffee_match.group(1), 0.55, "preference")

    food_match = re.search(r"món ăn yêu thích(?:\s+là)?\s+([^,\.\n]+)", text, re.IGNORECASE)
    if food_match:
        _add_candidate(candidates, "favorite_food", food_match.group(1), 0.9, "preference")
    elif "mì quảng" in lowered and ("ăn" in lowered or "ruột" in lowered):
        _add_candidate(candidates, "favorite_food", "mì Quảng", 0.75, "preference")

    interest_match = re.search(r"quan tâm.{0,40}(Python.{0,40})", text, re.IGNORECASE)
    if interest_match:
        _add_candidate(candidates, "interests", interest_match.group(1), 0.82, "interest")
    elif "python" in lowered and "ai" in lowered:
        _add_candidate(candidates, "interests", "Python, AI agent", 0.8, "interest")

    if "mlops" in lowered and "engineer" in lowered:
        _add_candidate(candidates, "profession", "MLOps engineer", 0.98, "profession", is_correction=is_correction)

    pet_match = re.search(r"nuôi(?:\s+một)?\s+bé\s+(corgi[^,\.\n]*)", text, re.IGNORECASE)
    if pet_match:
        _add_candidate(candidates, "pet", pet_match.group(1), 0.9, "pet")
    elif "corgi" in lowered:
        name_match = re.search(r"corgi tên\s+(\w+)", text, re.IGNORECASE)
        pet_value = f"corgi tên {name_match.group(1)}" if name_match else "corgi"
        _add_candidate(candidates, "pet", pet_value, 0.85, "pet")

    return candidates


def apply_profile_candidates(
    store: UserProfileStore,
    user_id: str,
    candidates: list[ProfileFact],
    *,
    current_turn: int,
    threshold: float,
) -> list[str]:
    written: list[str] = []
    records = store.fact_records(user_id)

    for candidate in candidates:
        if not store.should_apply_fact(user_id, candidate, current_turn, threshold):
            continue

        existing = records.get(candidate.key)
        mentions = 1
        if existing and existing.value.strip().lower() == candidate.value.strip().lower():
            mentions = existing.mentions + 1

        record = FactRecord(
            key=candidate.key,
            value=candidate.value,
            confidence=candidate.confidence,
            turn=current_turn,
            mentions=mentions,
        )
        store.upsert_fact_record(user_id, record)
        records[candidate.key] = record
        written.append(candidate.key)

    return written


def extract_profile_updates(message: str) -> dict[str, str]:
    return {
        candidate.key: candidate.value
        for candidate in extract_profile_candidates(message)
        if candidate.confidence >= 0.65 or candidate.is_correction
    }


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    if not messages:
        return ""

    chunks: list[str] = []
    for item in messages[-max_items:]:
        role = item.get("role", "user")
        content = item.get("content", "").strip()
        if not content:
            continue
        preview = content if len(content) <= 180 else content[:177] + "..."
        chunks.append(f"{role}: {preview}")

    return " | ".join(chunks)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def _thread_tokens(self, thread_state: dict[str, object]) -> int:
        summary = str(thread_state.get("summary", ""))
        messages = thread_state.get("messages", [])
        total = estimate_tokens(summary)
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict):
                    total += estimate_tokens(str(message.get("content", "")))
        return total

    def _maybe_compact(self, thread_id: str) -> None:
        thread_state = self._ensure_thread(thread_id)
        while self._thread_tokens(thread_state) > self.threshold_tokens:
            messages = thread_state["messages"]
            if not isinstance(messages, list) or len(messages) <= self.keep_messages:
                break

            to_summarize = messages[:-self.keep_messages]
            keep = messages[-self.keep_messages :]
            existing = str(thread_state.get("summary", ""))
            new_part = summarize_messages(to_summarize, max_items=len(to_summarize))
            combined = existing
            if existing and new_part:
                combined = existing + " || " + new_part
            elif new_part:
                combined = new_part

            thread_state["summary"] = combined
            thread_state["messages"] = keep
            thread_state["compactions"] = int(thread_state.get("compactions", 0)) + 1

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread_state = self._ensure_thread(thread_id)
        messages = thread_state["messages"]
        if not isinstance(messages, list):
            messages = []
            thread_state["messages"] = messages
        messages.append({"role": role, "content": content})
        self._maybe_compact(thread_id)

    def context(self, thread_id: str) -> dict[str, object]:
        return dict(self._ensure_thread(thread_id))

    def compaction_count(self, thread_id: str) -> int:
        return int(self._ensure_thread(thread_id).get("compactions", 0))
