# Lộ trình học AI/LLM từ repo này (cho dev Java Spring Boot)

> Viết cho người đã biết lập trình backend (Java/Spring), chưa có nền tảng AI.
> Mục tiêu: **hiểu LLM (ChatGPT, Claude…) hoạt động ra sao từ bên trong**, rồi **biết áp dụng vào sản phẩm web**.

---

## 0. Bức tranh tổng thể – bạn đang học cái gì?

Có hai hướng học AI, và bạn nên đi **cả hai, song song**:

| Hướng | Học cái gì | Ví dụ việc làm được | Repo này? |
|---|---|---|---|
| **A. Hiểu bên trong** (builder) | Tokenizer, embedding, attention, transformer, training, finetuning | Giải thích được vì sao model "bịa", vì sao context có giới hạn, tự finetune model nhỏ | ✅ Đây là trọng tâm repo |
| **B. Ứng dụng** (AI engineer) | Gọi API LLM, prompt, RAG, embeddings + vector DB, tool calling/agent | Chatbot hỏi đáp tài liệu công ty trong app Spring Boot | ❌ Học thêm bên ngoài (mục 6) |

Thực tế công việc của dev web dùng **B nhiều hơn**, nhưng có **A** thì bạn dùng B tốt hơn hẳn (debug được, chọn model đúng, ước lượng chi phí, hiểu giới hạn).

Repo này là code của sách **"Build a Large Language Model (From Scratch)" – Sebastian Raschka**. Bạn sẽ tự code một GPT nhỏ bằng PyTorch, train nó, load trọng số GPT-2 thật, rồi finetune thành bộ phân loại spam và chatbot làm theo lệnh.

> 💡 **Nên mua sách** (Manning/Amazon). Notebook trong repo có giải thích, nhưng sách có hình vẽ và lời giảng đầy đủ hơn nhiều. Nếu không mua, bạn vẫn học được với notebook + hỏi Claude.

---

## 1. Chuẩn bị (tuần 0, khoảng 3–5 buổi)

### 1.1 Python cho dev Java – học nhanh, chỉ cần đủ dùng

Bạn không cần học Python bài bản. Cần nắm:

| Java | Python |
|---|---|
| `List<Integer> a = new ArrayList<>()` | `a = []` |
| `Map<String,Integer>` | `d = {}` |
| `for (int i=0;i<n;i++)` | `for i in range(n):` |
| class + constructor | `class X:` + `def __init__(self, ...)` |
| `stream().map(...)` | list comprehension `[f(x) for x in a]` |
| Maven/Gradle | `pip` / `uv` + `requirements.txt` |
| `interface` / abstract | kế thừa `torch.nn.Module` |

Cần thêm: slicing (`a[1:3]`, `a[:, -1]`), unpacking (`b, t, d = x.shape`), `*args/**kwargs`, f-string.

### 1.2 NumPy/PyTorch tensor – quan trọng nhất

Tensor = mảng nhiều chiều. **90% lỗi khi học là sai shape.** Học thuộc thói quen: *luôn in `x.shape`*.

👉 Học **[appendix-A](appendix-A/)** (Introduction to PyTorch) **trước** chương 2. Đây là phần bắt buộc với bạn.

### 1.3 Toán – chỉ cần mức "trực giác"

- Nhân ma trận (matmul) – hiểu shape `(a,b) @ (b,c) -> (a,c)`
- Dot product = độ "giống nhau" giữa 2 vector
- Softmax – biến điểm số thành xác suất
- Đạo hàm/gradient ở mức ý tưởng: "hướng chỉnh trọng số để lỗi giảm"

Nguồn ngắn, trực quan: **3Blue1Brown** – series *Neural Networks* (YouTube, có phụ đề). Xem các video về neural network, backprop, transformer, attention. Không cần học giải tích bài bản.

### 1.4 Môi trường chạy code

Ba lựa chọn, theo thứ tự khuyên dùng:

1. **Google Colab** (miễn phí, có GPU): mở file `.ipynb` từ GitHub → chạy. Dễ nhất.
2. **Máy cá nhân**: theo [setup/README.md](setup/README.md). Chương 2–5 chạy được bằng CPU (chậm hơn nhưng ổn). Mac M-series dùng được `mps`.
3. **VS Code + Jupyter extension**: nếu quen IDE, mở notebook ngay trong VS Code.

```bash
# trên máy cá nhân, cách nhanh
pip install uv
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
jupyter lab
```

---

## 2. Lộ trình theo chương (khoảng 8–12 tuần, 5–8 giờ/tuần)

| Tuần | Nội dung | File chính | Bạn phải trả lời được |
|---|---|---|---|
| 0 | PyTorch cơ bản | `appendix-A/01_main-chapter-code/code-part1.ipynb`, `code-part2.ipynb` | Tensor, autograd, `nn.Module`, vòng lặp training gồm những bước nào? |
| 1 | LLM là gì | `ch01/README.md` + sách chương 1 | Pretraining vs finetuning? GPT khác BERT chỗ nào? |
| 2 | Xử lý văn bản | `ch02/01_main-chapter-code/ch02.ipynb` | Token là gì? BPE? Vì sao model "đếm chữ" kém? Embedding + positional embedding? |
| 3–4 | **Attention** (khó nhất) | `ch03/01_main-chapter-code/ch03.ipynb` | Q, K, V là gì? Vì sao chia √d? Causal mask để làm gì? Multi-head? |
| 5 | Kiến trúc GPT | `ch04/01_main-chapter-code/ch04.ipynb`, `gpt.py` | LayerNorm, GELU, shortcut, transformer block. Đếm được số tham số. Model sinh text từng token thế nào? |
| 6–7 | Pretraining | `ch05/01_main-chapter-code/ch05.ipynb` | Loss/cross-entropy, perplexity, temperature, top-k. Load trọng số GPT-2. |
| 8 | Finetune phân loại | `ch06/01_main-chapter-code/ch06.ipynb` | Biến GPT thành bộ phân loại spam. Vì sao chỉ train vài layer cuối? |
| 9–10 | Finetune làm theo lệnh | `ch07/01_main-chapter-code/ch07.ipynb` | Instruction dataset, prompt template, đánh giá model bằng model khác. |
| Thêm | LoRA | `appendix-E` | Finetune rẻ hơn bằng ma trận hạng thấp. |
| Thêm | Training tricks | `appendix-D` | Warmup, cosine decay, gradient clipping. |

**Sau khi xong lộ trình chính**, các thư mục bonus đáng xem:
- `ch04/03_kv-cache` – vì sao sinh text nhanh được (rất hay hỏi khi phỏng vấn)
- `ch04/04_gqa`, `ch04/07_moe` – kiến trúc của model hiện đại
- `ch05/07_gpt_to_llama`, `ch05/11_qwen3` – từ GPT-2 lên Llama/Qwen thật
- `ch07/04_preference-tuning-with-dpo` – cách "dạy" model trả lời đúng ý người dùng

⚠️ **Đừng nhảy cóc chương 3.** Nếu attention chưa hiểu, mọi thứ phía sau sẽ mơ hồ. Dành gấp đôi thời gian cho nó cũng xứng đáng.

---

## 3. Cách học một chương (quy trình 5 bước)

Đọc hiểu ≠ biết. Với dev, cách học hiệu quả nhất là **tự gõ lại code**.

1. **Đọc lướt** (30'): đọc sách/notebook, chưa chạy. Ghi ra 3–5 câu hỏi.
2. **Chạy notebook** (1h): chạy từng cell. Trước mỗi cell, **đoán output/shape**, rồi chạy để kiểm tra.
3. **Tự gõ lại** (2–3h): mở notebook trống, gõ lại phần chính **không copy**. Bí thì nhìn, nhưng gõ tay.
4. **Phá code** (30'): sửa thử – bỏ mask, đổi số head, bỏ chia √d, đổi temperature… xem chuyện gì xảy ra.
5. **Giải thích lại** (30'): viết 5–10 dòng vào sổ ghi chú bằng lời của bạn, hoặc giải thích cho Claude và nhờ chấm.

Làm bài tập trong `exercise-solutions.ipynb` **sau khi tự thử**, không phải trước.

### Sổ ghi chú học tập

Tạo một file `notes/` riêng (hoặc Notion), mỗi chương 1 file:

```markdown
# Chương 3 – Attention
## Tóm tắt bằng lời của tôi
## Shape của các tensor chính
## Chỗ tôi từng hiểu sai
## Câu hỏi còn mở
```

---

## 4. Cách dùng Claude để học hiệu quả

Claude không chỉ để "hỏi đáp". Dùng như **gia sư + reviewer + bạn pair-programming**.

### 4.1 Những kiểu prompt nên dùng

**Giải thích theo ngôn ngữ của bạn**
> "Giải thích multi-head attention trong `ch03/01_main-chapter-code/ch03.ipynb` cho dev Java Spring. Dùng ẩn dụ backend nếu được, kèm shape tensor từng bước."

**Giải thích một đoạn code cụ thể**
> "Đọc class `MultiHeadAttention` trong `ch04/01_main-chapter-code/gpt.py`, giải thích từng dòng trong `forward`, ghi shape sau mỗi dòng với batch=2, seq=4, d=768, heads=12."

**Kiểm tra mức hiểu (rất hiệu quả)**
> "Hỏi tôi 5 câu về chương 2 từ dễ đến khó. Mỗi lần một câu, chờ tôi trả lời rồi chấm và giải thích chỗ sai."

**Feynman – bạn giảng, Claude bắt lỗi**
> "Tôi sẽ giải thích causal mask, bạn chỉ ra chỗ tôi sai hoặc mơ hồ: …"

**Review code bạn tự gõ**
> "Tôi tự viết lại `SelfAttention` ở file X. So với bản gốc trong sách, chỉ ra lỗi và chỗ khác biệt, đừng sửa hộ."

**Debug lỗi shape**
> "Lỗi `RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x768 and 512x768)` khi chạy cell này: … Giải thích nguyên nhân trước, gợi ý cách sửa sau."

**Liên hệ thực tế**
> "Khái niệm KV cache trong `ch04/03_kv-cache` ảnh hưởng thế nào đến chi phí và độ trễ khi tôi gọi API LLM từ Spring Boot?"

**Lên kế hoạch / ôn tập**
> "Tôi vừa xong chương 4. Tóm tắt những gì tôi cần nhớ trước khi sang chương 5, và 3 bài tập nhỏ tự làm."

### 4.2 Nguyên tắc dùng AI khi học

- ❌ Đừng nhờ "viết code hộ" phần bạn đang học. Bạn sẽ có code chạy được nhưng không hiểu.
- ✅ Nhờ **giải thích, gợi ý, review, ra đề, chấm bài**.
- ✅ Khi Claude giải thích, **luôn kiểm chứng bằng cách chạy code**. AI cũng có thể sai.
- ✅ Hỏi "tại sao" nhiều hơn "làm thế nào".
- ✅ Ghi rõ đường dẫn file khi hỏi (trong repo này Claude đọc được file trực tiếp).

### 4.3 Dùng Claude Code (công cụ đang dùng) trong repo này

- Session cloud: Claude đọc được toàn bộ repo, chạy được Python, tạo file, commit/push lên GitHub của bạn.
- Bạn có thể nói: *"Tạo notebook `notes/ch03_practice.ipynb` có các ô trống để tôi tự điền phần attention, kèm test kiểm tra shape"* → Claude tạo bài tập, bạn làm, rồi nhờ chấm.
- Cài Claude Code trên máy (CLI hoặc extension VS Code/IntelliJ) để học ngay trong IDE quen thuộc.

---

## 5. Kiểm tra bạn đã thật sự hiểu (checklist)

Sau toàn bộ lộ trình, bạn nên:

- [ ] Vẽ được sơ đồ GPT: token → embedding → N × (attention + FFN) → logits → token tiếp theo
- [ ] Tự viết `SelfAttention` có causal mask mà không nhìn sách
- [ ] Đếm được số tham số GPT-2 small (~124M) và ước lượng bộ nhớ cần
- [ ] Giải thích temperature, top-k, top-p ảnh hưởng output thế nào
- [ ] Giải thích vì sao context window có giới hạn và chi phí tăng theo độ dài
- [ ] Giải thích pretraining vs SFT vs DPO/RLHF
- [ ] Finetune được GPT-2 cho một bài toán phân loại của riêng bạn (ví dụ: phân loại ticket hỗ trợ)

---

## 6. Học thêm ngoài repo (hướng B – áp dụng vào web)

Sau khoảng chương 5 có thể bắt đầu song song:

1. **Gọi API LLM từ Java**: Anthropic API / OpenAI API – thử gọi bằng HTTP client thuần trước, rồi mới dùng thư viện.
2. **Spring AI** (`spring-projects/spring-ai`): framework chính thức của Spring cho LLM, embedding, vector store. Rất hợp với bạn.
   (LangChain4j là lựa chọn thay thế.)
3. **Embeddings + RAG**: chia tài liệu → embedding → lưu vector DB (pgvector trên PostgreSQL là dễ nhất) → truy vấn → nhét vào prompt. Bạn sẽ thấy embedding ở chương 2 chính là nền tảng.
4. **Tool calling / Agent**: cho LLM gọi API nội bộ (ví dụ "tra cứu đơn hàng") – nối thẳng với kinh nghiệm REST của bạn.
5. **Đánh giá (eval)**: làm sao biết chatbot trả lời đúng – liên quan `ch07/03_model-evaluation`.

**Project tốt nghiệp gợi ý**: một app Spring Boot có
- chatbot hỏi đáp tài liệu nội bộ (RAG + pgvector),
- một endpoint phân loại nội dung dùng model nhỏ bạn tự finetune (từ ch06), phục vụ qua một service Python (FastAPI) nhỏ.

Project này kết hợp được cả hướng A và B, rất đẹp để đưa vào CV.

### Tài nguyên bổ sung

- 3Blue1Brown – *Neural Networks* (trực giác toán)
- Andrej Karpathy – *Neural Networks: Zero to Hero*, đặc biệt video "Let's build GPT" (YouTube) – bổ trợ cực tốt cho chương 3–5
- `ch01/reading-recommendations.md` trong repo này
- Sách tiếp theo của cùng tác giả: *Build a Reasoning Model (From Scratch)*

---

## 7. Lịch gợi ý mỗi tuần

| Buổi | Việc |
|---|---|
| Buổi 1 (1–1.5h) | Đọc + chạy notebook (bước 1–2) |
| Buổi 2 (1.5–2h) | Tự gõ lại code (bước 3) |
| Buổi 3 (1h) | Phá code + bài tập (bước 4) |
| Cuối tuần (30') | Viết ghi chú, nhờ Claude quiz 5 câu (bước 5) |

Đều đặn quan trọng hơn cường độ: **4 buổi/tuần × 1h** tốt hơn 1 buổi 6h.

---

## 8. Theo dõi tiến độ

- [ ] Tuần 0 – Python + PyTorch (appendix A)
- [ ] Chương 1 – Tổng quan LLM
- [ ] Chương 2 – Tokenizer & embedding
- [ ] Chương 3 – Attention
- [ ] Chương 4 – Kiến trúc GPT
- [ ] Chương 5 – Pretraining & load GPT-2
- [ ] Chương 6 – Finetune phân loại
- [ ] Chương 7 – Instruction finetuning
- [ ] Appendix D, E – Training tricks & LoRA
- [ ] Bắt đầu hướng B: Spring AI + RAG
- [ ] Project tốt nghiệp
