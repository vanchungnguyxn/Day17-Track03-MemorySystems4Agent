# Phân tích kết quả benchmark — Day 17 Memory Systems

## Vì sao Advanced có recall tốt hơn Baseline?

Baseline chỉ giữ message trong cùng một `thread_id`. Khi benchmark hỏi lại ở thread mới (`conv-01-recall`), session cũ không còn trong prompt nên agent không thể nhớ tên, nơi ở hay preference.

Advanced ghi các fact ổn định vào `User.md` qua pipeline bonus `extract_profile_candidates()` → `apply_profile_candidates()`. Khi sang thread mới, profile vẫn được inject vào ngữ cảnh, nên cross-session recall cao hơn rõ rệt (benchmark ~0.82 vs 0.00).

## Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Mỗi lượt Advanced phải đọc thêm `User.md`, chấm confidence, cập nhật metadata, và duy trì compact state. Ở standard benchmark, prompt tokens Advanced (~20k) cao hơn Baseline (~16k) vì overhead persistent memory chưa được bù bằng tiết kiệm compact (chuỗi chưa đủ dài để compact kích hoạt).

## Vì sao compact giúp Advanced ở hội thoại dài?

Trong stress benchmark, Baseline giữ toàn bộ lịch sử message nên prompt tokens tăng mạnh (~23.7k). Advanced nén phần cũ sang summary và chỉ giữ message gần nhất, kéo prompt xuống ~14.6k với ~21 lần compaction, trong khi recall vẫn ~0.83 nhờ facts cốt lõi nằm trong `User.md`.

## Memory growth và rủi ro

`User.md` tăng theo số fact mới. Rủi ro chính: lưu sai fact, conflict khi user sửa thông tin, file phình to theo thời gian.

---

## Bonus: Guardrails cho persistent memory

Lab bổ sung bốn lớp bảo vệ để hướng tới rubric 90–100.

### 1. Confidence threshold (`PROFILE_CONFIDENCE_THRESHOLD`, mặc định 0.65)

Mỗi fact được trích thành `ProfileFact` kèm `confidence` và `entity_type`. Chỉ fact đạt ngưỡng mới ghi vào `User.md`.

| Nguồn | Confidence | Ghi? |
|-------|------------|------|
| `mình tên là DũngCT` | 0.95 | Có |
| `đồ uống yêu thích là cà phê sữa đá` | 0.90 | Có |
| `uống cà phê` (suy luận yếu) | 0.55 | Không |

**Lợi ích:** giảm pollution trong `User.md`, recall ổn định hơn.

**Rủi ro:** fact đúng nhưng diễn đạt mơ hồ có thể bị bỏ qua → cần user nhắc lại hoặc hạ ngưỡng.

### 2. Conflict handling (correction ưu tiên)

Khi message có marker đính chính (`đính chính`, `chuyển sang`, `không còn`…), fact được đánh dấu `is_correction=True` và **luôn ghi đè** fact cũ cùng key nếu confidence ≥ 0.5 — kể cả khi fact mới có confidence thấp hơn bản cũ.

Ví dụ benchmark: `backend engineer` → `MLOps engineer`, `Đà Nẵng` → `Huế`.

**Lợi ích:** không giữ đồng thời hai nghề/nơi ở mâu thuẫn.

**Rủi ro:** correction giả (user đùa) vẫn có thể ghi đè nếu vượt qua noise filter.

### 3. Entity extraction có cấu trúc

`ProfileFact` tách field theo loại: `person`, `location`, `profession`, `preference`, `interest`, `pet`. Metadata lưu trong `User.md`:

```markdown
- profession: MLOps engineer <!-- meta:conf=0.98,turn=12,mentions=2 -->
```

**Lợi ích:** dễ debug, benchmark được confidence/turn/mentions, conflict resolution rõ ràng.

### 4. Memory decay (`MEMORY_DECAY_TURNS`, mặc định 25)

Fact thuộc nhóm phụ (`interests`, `favorite_food`, `favorite_drink`, `pet`) sẽ **không còn active** khi:
- đã quá `memory_decay_turns` kể từ lần cập nhật, và
- `mentions < 2` (chưa được nhắc lại).

Fact cốt lõi (`name`, `location`, `profession`, `response_style`) **không decay**.

**Lợi ích:** prompt gọn hơn theo thời gian, tránh nhồi preference cũ ít dùng.

**Rủi ro:** fact phụ quan trọng nhưng chỉ nhắc một lần có thể biến mất khỏi recall active — cần cân `decay_turns` hoặc tăng `mentions` khi user nhắc lại.

### 5. Chặn câu hỏi và nhiễu

- `is_question_only_turn()`: không ghi câu hỏi recall vào profile.
- `NOISE_MARKERS`: bỏ qua turn chứa nhiễu (`chỉ là nơi… họp`, `câu đùa`…).

---

## Trade-off tổng quan

| Khía cạnh | Baseline | Advanced | Advanced + Bonus |
|-----------|----------|----------|------------------|
| Cross-session recall | Thấp | Cao | Cao, ít fact sai hơn |
| Prompt cost (ngắn) | Thấp | Cao hơn | Cao hơn một chút (metadata) |
| Prompt cost (dài) | Tăng mạnh | Ổn định nhờ compact | Ổn định + decay fact phụ |
| Độ phức tạp | Đơn giản | Trung bình | Cao hơn, cần tune threshold/decay |
| Rủi ro lưu sai | Không persistent | Trung bình | Thấp hơn nhờ guardrail |

Hệ thống mạnh hơn nhưng phức tạp hơn: persistent memory cho recall, compact memory cho chi phí ngữ cảnh, bonus guardrail, và **live LangGraph runtime** khi có API key.

---

## Live LangGraph runtime (Guide Bước 4–5)

`agent_runtime.py` wire agent production-style:

| Agent | Thành phần |
|-------|------------|
| Baseline | `create_agent` + `InMemorySaver`, không tools, không User.md |
| Advanced | `InMemorySaver` + profile tools + `SummarizationMiddleware` |

**Lợi ích:** benchmark offline vẫn deterministic; khi bật live mode, cùng kiến trúc memory có thể gọi LLM thật.

**Rủi ro:** live mode phụ thuộc API key/latency; summarization middleware tốn thêm token khi trigger.
