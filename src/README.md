# Day 17 — Memory Systems for AI Agent

**Giai đoạn 2 · Track 3 · Day 17**

## Thông tin sinh viên

| | |
|---|---|
| **Họ và tên** | Hai Dương |
| **MSSV** | _[Điền MSSV]_ |
| **Môn / Track** | Memory Systems for AI Agent |
| **Ngày nộp** | 19/06/2026 |

> Nếu tên hoặc MSSV khác, sửa trực tiếp hai dòng trên trước khi nộp bài.

---

## Tóm tắt bài làm

Bài lab xây dựng và so sánh hai agent trên cùng benchmark tiếng Việt:

| Agent | Memory | Mục đích |
|-------|--------|----------|
| **Baseline** | Chỉ short-term trong cùng `thread_id` | Mốc so sánh “ngây thơ”, không nhớ cross-session |
| **Advanced** | Short-term + `User.md` + compact memory | Nhớ facts bền vững, tối ưu prompt khi hội thoại dài |

Ba lớp memory của Advanced:

1. **Short-term** — message gần nhất trong thread (compact manager / LangGraph checkpointer)
2. **Persistent** — facts ổn định trong `state/profiles/<user>/User.md`
3. **Compact** — nén lịch sử cũ thành summary khi vượt ngưỡng token

---

## Cấu trúc mã nguồn

| File | Vai trò |
|------|---------|
| `config.py` | `LabConfig`, paths, compact threshold, provider, bonus settings |
| `model_provider.py` | 6 provider: openai, custom, gemini, anthropic, ollama, openrouter |
| `memory_store.py` | Token estimate, `UserProfileStore`, extract facts, compact memory, bonus guardrails |
| `agent_baseline.py` | Baseline agent (offline + live) |
| `agent_advanced.py` | Advanced agent (offline + live) |
| `agent_runtime.py` | LangGraph: `create_agent` + `InMemorySaver` + tools + summarization |
| `benchmark.py` | Standard + Long-context stress benchmark |
| `test_agents.py` | 9 tests (4 cốt lõi + 4 bonus + 1 LangGraph) |
| `ANALYSIS.md` | Phân tích trade-off và bonus |
| `verify_openai.py` | Kiểm tra kết nối OpenAI (tùy chọn) |

Dữ liệu benchmark: `../data/conversations.json`, `../data/advanced_long_context.json`

---

## Cài đặt và chạy

### 1. Môi trường

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest
```

### 2. Test (bắt buộc cho rubric)

```bash
cd src
python -m pytest test_agents.py -v
```

Kỳ vọng: **9/9 passed**

### 3. Benchmark (bắt buộc cho rubric)

```bash
cd src
python benchmark.py
```

Benchmark chạy **offline** (`force_offline=True`) — ổn định, không cần API key, đúng thiết kế chấm điểm.

### 4. OpenAI live (tùy chọn, không bắt buộc nộp)

Tạo `.env` ở root repo:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

```bash
cd src
python verify_openai.py
```

---

## Kết quả benchmark (offline)

Chạy trên máy local, có thể lệch nhẹ giữa các lần chạy:

### Standard Benchmark

| Agent | Agent tokens | Prompt tokens | Recall | Quality | Memory (B) | Compactions |
|-------|-------------|---------------|--------|---------|------------|-------------|
| Baseline | ~1806 | ~16152 | **0.00** | ~0.10 | 0 | 0 |
| Advanced | ~1180 | ~28000 | **~0.75** | ~0.88 | ~9 | 0 |

### Long-Context Stress Benchmark

| Agent | Agent tokens | Prompt tokens | Recall | Quality | Memory (B) | Compactions |
|-------|-------------|---------------|--------|---------|------------|-------------|
| Baseline | ~483 | ~23770 | **0.00** | ~0.10 | 0 | 0 |
| Advanced | ~353 | ~15400 | **~0.67** | ~0.83 | ~0–150 | **~21** |

**Nhận xét ngắn:**

- Advanced recall cao hơn rõ nhờ `User.md`
- Hội thoại ngắn: Advanced có thể tốn prompt hơn Baseline (overhead persistent memory)
- Hội thoại dài: compact giúp Advanced prompt thấp hơn Baseline ~35%
- Chi tiết: xem `ANALYSIS.md`

---

## Bonus (Rubric 90–100)

| Tính năng | Mô tả |
|-----------|--------|
| **Confidence threshold** | Chỉ ghi `User.md` khi `confidence ≥ 0.65` |
| **Conflict handling** | Correction (`đính chính`, `chuyển sang`) ghi đè fact cũ |
| **Entity extraction** | `ProfileFact` + metadata `conf/turn/mentions` trong User.md |
| **Memory decay** | Fact phụ (`interests`, food/drink/pet) ẩn khi quá cũ và ít được nhắc lại |
| **Question/noise filter** | Không ghi câu hỏi recall hoặc nhiễu vào profile |

Test bonus: `test_confidence_threshold_*`, `test_correction_*`, `test_noise_*`, `test_memory_decay_*`

---

## Live LangGraph mode (tùy chọn)

Khi có API key, agent build qua `agent_runtime.py`:

- **Baseline**: `create_agent` + `InMemorySaver`, không tools
- **Advanced**: tools `read_user_profile`, `update_user_fact`, `edit_user_profile` + `SummarizationMiddleware`

```python
from agent_advanced import AdvancedAgent

agent = AdvancedAgent(force_offline=False)
result = agent.reply("dungct", "thread-1", "Chào bạn, mình tên là DũngCT.")
print(result["mode"], result["answer"])
```

---

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `LLM_PROVIDER` | `openai` | Provider chính |
| `LLM_MODEL` | `gpt-4o-mini` | Model chính |
| `COMPACT_THRESHOLD_TOKENS` | `800` | Ngưỡng kích hoạt compact |
| `COMPACT_KEEP_MESSAGES` | `6` | Số message giữ lại sau compact |
| `PROFILE_CONFIDENCE_THRESHOLD` | `0.65` | Ngưỡng ghi User.md |
| `MEMORY_DECAY_TURNS` | `25` | Turn trước khi decay fact phụ |

---

## Tài liệu liên quan

- [`../Guide.md`](../Guide.md) — hướng dẫn từng bước
- [`../Rubric.md`](../Rubric.md) — tiêu chí chấm
- [`ANALYSIS.md`](ANALYSIS.md) — phân tích kết quả và trade-off bonus
