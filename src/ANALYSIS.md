# Phân tích kết quả benchmark — Day 17 Memory Systems

> Số liệu dưới đây lấy từ lần chạy `python benchmark.py` gần nhất (offline mode).

## Kết quả benchmark

### Standard Benchmark (`data/conversations.json`)

| Agent | Agent tokens | Prompt tokens | Recall | Quality | Memory (B) | Compactions |
|-------|-------------|---------------|--------|---------|------------|-------------|
| Baseline | 1806 | 16152 | **0.00** | 0.10 | 0 | 0 |
| Advanced | 1180 | 28280 | **0.75** | 0.88 | 9 | 0 |

### Long-Context Stress Benchmark (`data/advanced_long_context.json`)

| Agent | Agent tokens | Prompt tokens | Recall | Quality | Memory (B) | Compactions |
|-------|-------------|---------------|--------|---------|------------|-------------|
| Baseline | 483 | 23770 | **0.00** | 0.10 | 0 | 0 |
| Advanced | 353 | 15394 | **0.67** | 0.83 | 0 | **21** |

---

## Câu chuyện 5 bước (logic Rubric)

### 1. Baseline không nhớ dài hạn

Baseline chỉ giữ message trong cùng `thread_id`, không có `User.md`. Khi benchmark hỏi recall ở thread mới (`conv-01-recall`), session cũ không còn trong ngữ cảnh.

**Chứng minh:** cross-session recall Baseline = **0.00**; test `test_cross_session_recall` xác nhận Baseline không trả lời đúng tên ở thread mới.

### 2. Advanced thêm User.md nên recall tăng

Advanced trích fact ổn định qua `extract_profile_candidates()` → `apply_profile_candidates()` và lưu vào `User.md`. Sang thread mới, profile vẫn được đọc vào ngữ cảnh.

**Chứng minh:** recall Advanced = **0.75** (standard) và **0.67** (stress) so với **0.00** của Baseline.

### 3. Hội thoại dài làm prompt cost tăng mạnh

Baseline giữ toàn bộ lịch sử message trong session. Càng nhiều turn, prompt tokens càng tăng tuyến tính.

**Chứng minh:** stress benchmark — Baseline prompt = **23770** tokens.

### 4. Compact memory kéo chi phí ngữ cảnh xuống

Advanced chuyển message cũ sang `summary` khi vượt `compact_threshold_tokens`, chỉ giữ `compact_keep_messages` message gần nhất. Facts cốt lõi vẫn nằm trong `User.md`.

**Chứng minh:** stress benchmark — Advanced prompt = **15394** (giảm ~**35%** so với Baseline), **21 compactions**; test `test_compact_reduces_prompt_load_on_long_thread` xác nhận prompt Advanced < Baseline trên thread dài.

### 5. Hệ thống mạnh hơn nhưng phức tạp hơn — cần guardrail

Advanced thêm persistent memory, compact logic, metadata, và bonus guardrails. Trade-off: recall tốt hơn và prompt ổn định hơn ở hội thoại dài, nhưng overhead cao hơn ở hội thoại ngắn và cần tune threshold/decay.

---

## Vì sao Advanced có recall tốt hơn Baseline?

Baseline quên cross-session vì không có lớp persistent. Advanced ghi fact vào `User.md` nên vẫn trả lời được câu hỏi recall ở thread mới — recall **0.75 / 0.67** vs **0.00**.

Recall chưa đạt 1.0 vì extraction dùng regex heuristic: một số câu hỏi recall đa fact (tên + đồ uống + style) chỉ match được một phần `expected_contains`.

## Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Mỗi lượt Advanced phải đọc/cập nhật `User.md`, chấm confidence, ghi metadata, và duy trì compact state. Standard benchmark chưa đủ dài để compact kích hoạt (0 compactions).

**Chứng minh:** standard — Advanced prompt **28280** > Baseline **16152** (~+75% overhead persistent memory).

## Vì sao compact giúp Advanced ở hội thoại dài?

Compact nén lịch sử cũ thành summary, giữ message gần nhất. Baseline vẫn mang toàn bộ lịch sử → prompt phình to.

**Chứng minh:** stress — Advanced **15394** vs Baseline **23770** prompt tokens; recall vẫn **0.67** nhờ facts trong `User.md`.

## Memory growth và rủi ro chung

`User.md` tăng theo số fact mới. Rủi ro: lưu sai fact, conflict chưa xử lý đúng, file phình to nếu ghi quá nhiều chi tiết tạm thời. Bonus guardrails (dưới đây) giảm các rủi ro này.

---

## Bonus: Guardrails cho persistent memory

Mỗi bonus được mô tả theo 3 câu hỏi Rubric: **giải quyết vấn đề gì → cải thiện recall/token thế nào → rủi ro gì**.

### 1. Confidence threshold (`PROFILE_CONFIDENCE_THRESHOLD` = 0.65)

**Vấn đề giải quyết:** Trích xuất heuristic dễ ghi fact suy luận yếu vào `User.md` (ví dụ chỉ vì user nhắc "uống cà phê" mà ghi nhầm `favorite_drink`).

**Cải thiện recall / token:**
- **Recall:** fact trong profile đáng tin hơn → ít trả lời sai do pollution (ví dụ `"gì?"`, `"của mình"` không còn bị ghi thành tên).
- **Token:** User.md gọn hơn → mỗi lượt đọc profile tốn ít prompt hơn về lâu dài.

| Nguồn | Confidence | Ghi? |
|-------|------------|------|
| `mình tên là DũngCT` | 0.95 | Có |
| `đồ uống yêu thích là cà phê sữa đá` | 0.90 | Có |
| `uống cà phê` (suy luận yếu) | 0.55 | Không |

**Rủi ro:** Fact đúng nhưng diễn đạt mơ hồ có thể bị bỏ qua → recall giảm nếu user không nhắc lại rõ ràng. Cần cân `PROFILE_CONFIDENCE_THRESHOLD`.

**Test:** `test_confidence_threshold_blocks_weak_inference`

---

### 2. Conflict handling (correction ưu tiên)

**Vấn đề giải quyết:** Dataset có correction (`backend engineer` → `MLOps engineer`, `Đà Nẵng` → `Huế`). Nếu append fact mới mà không ghi đè, profile giữ đồng thời fact cũ và mới → recall sai.

**Cải thiện recall / token:**
- **Recall:** Correction (`is_correction=True`) luôn ghi đè fact cùng key → trả lời đúng thông tin mới nhất (conv-03, conv-06, stress test).
- **Token:** Không tăng thêm vì chỉ thay một dòng trong User.md thay vì append trùng.

**Rủi ro:** Correction giả hoặc câu đùa có marker "chuyển sang" vẫn có thể ghi đè fact thật nếu lọt qua noise filter.

**Test:** `test_correction_overrides_old_fact`

---

### 3. Entity extraction có cấu trúc (`ProfileFact` + metadata)

**Vấn đề giải quyết:** Lưu profile dạng text tự do khó debug, khó biết fact nào cũ/mới, khó resolve conflict theo key (`name`, `location`, `profession`…).

**Cải thiện recall / token:**
- **Recall:** Mỗi fact có `key` + `entity_type` + metadata (`conf`, `turn`, `mentions`) → conflict handling và memory decay hoạt động chính xác theo từng field, giúp recall ổn định qua nhiều phiên.
- **Token:** Metadata nhỏ so với lợi ích tránh duplicate fact; decay có thể ẩn fact phụ khỏi active context → giảm prompt khi hội thoại dài.

```markdown
- profession: MLOps engineer <!-- meta:conf=0.98,turn=12,mentions=2 -->
```

**Rủi ro:**
- User.md dài hơn một chút do metadata HTML comment → prompt tăng nhẹ mỗi lượt.
- Parser phụ thuộc format `- key: value` — nếu format lệch, `facts()` có thể đọc sai.
- Regex extraction vẫn không linh hoạt bằng LLM entity extraction thật.

---

### 4. Memory decay (`MEMORY_DECAY_TURNS` = 25)

**Vấn đề giải quyết:** Fact phụ (`interests`, food/drink/pet) nhắc một lần rồi không dùng lại vẫn nằm trong active context → prompt và User.md phình dần.

**Cải thiện recall / token:**
- **Token:** Fact phụ quá cũ và `mentions < 2` bị ẩn khỏi `_active_facts()` → prompt context gọn hơn ở hội thoại dài.
- **Recall:** Fact cốt lõi (`name`, `location`, `profession`, `response_style`) **không decay** → recall cross-session vẫn cao (**0.75 / 0.67**).

**Rủi ro:** Fact phụ quan trọng nhưng chỉ nhắc một lần (ví dụ `pet`) có thể biến mất khỏi active recall — user hỏi lại sau nhiều turn có thể không được trả lời. Giảm bằng cách tăng `mentions` khi user nhắc lại hoặc tăng `MEMORY_DECAY_TURNS`.

**Test:** `test_memory_decay_hides_stale_secondary_fact`

---

### 5. Chặn câu hỏi và nhiễu (`is_question_only_turn` + `NOISE_MARKERS`)

**Vấn đề giải quyết:**
- **Câu hỏi recall** ("Mình tên gì?", "Tên mình là gì và mình thích kiểu trả lời như thế nào?") bị regex trích nhầm thành fact → corrupt `User.md` (đã xảy ra khi chưa có guardrail: name = `"gì?"`, location = `"đâu?"`).
- **Turn nhiễu** ("Hà Nội chỉ là nơi bay ra họp", "đùa chuyển sang product manager") ghi đè fact thật.

**Cải thiện recall / token:**
- **Recall:** Profile không bị ghi đè bởi câu hỏi → recall cross-session ổn định hơn (đóng góp vào **0.75** standard thay vì ~0.32 trước khi fix).
- **Token:** Không ghi rác vào User.md → file nhỏ hơn, ít prompt overhead mỗi lượt.

**Rủi ro:**
- Declarative question ("Bạn có thể nhắc lại tên mình không?") vẫn có thể bị skip nếu không match `_DECLARATIVE_MARKERS` — miss cơ hội ghi fact từ câu hỏi có chứa thông tin mới.
- Noise filter quá aggressive có thể bỏ qua turn hợp lệ nếu chứa từ khóa nhiễu trong cùng câu với fact thật.

**Test:** `test_cross_session_recall`, `test_noise_location_is_ignored`

---

## Trade-off tổng quan

| Khía cạnh | Baseline | Advanced | Advanced + Bonus |
|-----------|----------|----------|------------------|
| Cross-session recall | **0.00** | **0.75 / 0.67** | Cao, ít fact sai hơn |
| Prompt cost (ngắn) | **16152** | **28280** | Cao hơn (metadata + profile) |
| Prompt cost (dài) | **23770** | **15394** | Thấp hơn ~35% + decay |
| Độ phức tạp | Đơn giản | Trung bình | Cao, cần tune threshold/decay |
| Rủi ro lưu sai | Không persistent | Trung bình | Thấp hơn nhờ guardrail |

---

## Live LangGraph runtime (tùy chọn, không bắt buộc chấm điểm)

`agent_runtime.py` wire agent production-style khi có API key:

| Agent | Thành phần |
|-------|------------|
| Baseline | `create_agent` + `InMemorySaver`, không tools |
| Advanced | `InMemorySaver` + profile tools + `SummarizationMiddleware` |

**Vấn đề giải quyết:** Demo agent với LLM thật thay vì chỉ offline heuristic.

**Cải thiện recall / token:** Live mode có thể trả lời tự nhiên hơn; summarization middleware giảm context dài trong LangGraph thread.

**Rủi ro:** Phụ thuộc API key, latency, chi phí token thật; kết quả không deterministic như offline benchmark.

**Test:** `test_langgraph_agents_build_and_invoke`
