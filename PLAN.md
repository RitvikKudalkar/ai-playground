# Learning Plan: Steps 2–5

Step 1 (Hello API) is done: we can call Gemini, use a system instruction, and hold a chat.
Each step below gets its own folder, builds on the previous one, and ends with a
"Done when" check so we know it's time to move on.

Default chat model: `gemini-3.5-flash-lite` (set in `.env`). Embedding model: `gemini-embedding-001`.

---

## Step 2 — Prompt Lab (`02-prompt-lab/`)

**Goal:** Learn how far you can push a model *without* changing it — and how to measure it.

**Concepts**
- System instructions: role, rules, tone, output length
- Zero-shot vs few-shot (giving 3–5 examples in the prompt)
- Structured output: forcing JSON with a schema (`response_schema` + Pydantic)
- Temperature and its effect on consistency
- Simple evaluation: run the same prompts against several variants and compare

**What we'll build**
| File | Does |
|---|---|
| `zero_vs_few_shot.py` | Classifies customer messages (complaint / question / praise) with 0 vs 3 examples; prints accuracy |
| `structured_output.py` | Extracts name, date, amount from messy text into a typed Python object |
| `prompt_compare.py` | Runs one test set through 3 system prompts side by side |
| `test_cases.json` | Small labeled dataset (~15 examples) used for scoring |

**Exercises**
- Write a system prompt that makes the model refuse off-topic questions.
- Find a case zero-shot gets wrong and fix it with a single example.

**Done when:** you can show a measurable accuracy jump from a prompt change, and get valid JSON back every time.

---

## Step 3 — Mini RAG: chat with your notes/PDFs (`03-rag/`)

**Goal:** Make the model answer from *your* documents, with sources, instead of its general knowledge.

**Concepts**
- Why RAG: models don't know your private data and can't be "taught" facts reliably via prompts
- Chunking: splitting documents into overlapping pieces
- Embeddings: turning text into vectors that capture meaning
- Vector search: cosine similarity → top-k most relevant chunks
- Grounded prompting: "answer only from this context, cite the source, say 'I don't know' otherwise"

**What we'll build** (in two passes, so you see the internals before using a library)
| File | Does |
|---|---|
| `docs/` | A few of your own PDFs / notes to search over |
| `rag_from_scratch.py` | Pass 1: chunk → embed with `gemini-embedding-001` → store in a NumPy array → cosine search → answer. ~100 lines, no magic |
| `ingest.py` + `chat.py` | Pass 2: same pipeline using **ChromaDB** so embeddings persist on disk, plus PDF loading with `pypdf` |
| `eval_questions.json` | 10 questions with known answers to check retrieval quality |

**New dependencies:** `numpy`, `pypdf`, `chromadb`

**Exercises**
- Change chunk size (200 vs 1000 chars) and see how answers change.
- Ask something not in the docs — confirm it says "I don't know" instead of making things up.

**Done when:** you can ask questions about your own documents and get correct answers with the source file/page cited.

---

## Step 4 — Tool Calling & a Simple Agent (`04-tools/`)

**Goal:** Let the model *do things* — call Python functions you wrote — instead of only producing text.

**Concepts**
- Function declarations: describing a function (name, params, docstring) so the model knows when to use it
- The tool loop: model asks to call a tool → your code runs it → result goes back → model answers
- Automatic vs manual function calling in the `google-genai` SDK
- Agents = tool loop + multiple tools + the model deciding the order
- Safety: never let the model run arbitrary code or destructive actions without checks

**What we'll build**
| File | Does |
|---|---|
| `calculator_tool.py` | Simplest case: one `add`/`multiply` tool, using the SDK's automatic calling |
| `manual_tool_loop.py` | Same thing written by hand, so you see every request/response in the loop |
| `assistant_agent.py` | A small agent with 3 tools: get weather (free API), read a local todo file, add a todo |
| `rag_tool.py` | Bonus: wraps Step 3's search as a tool, so the model decides *when* to look in your docs |

**Exercises**
- Ask a question that needs two tools in sequence and watch the loop.
- Give a tool a vague docstring and see the model misuse it — then fix the description.

**Done when:** the agent correctly chooses between tools (or no tool) for a set of 10 varied requests.

---

## Step 5 — Fine-Tuning (`05-finetune/`)

**Goal:** Actually change a model's weights, and understand when that's worth it versus Steps 2–4.

**Which route:** Fine-tuning Gemini itself needs Google Cloud **Vertex AI** with billing enabled. For learning,
we'll fine-tune **Gemma** — Google's open-weight model family, a sibling of Gemini — for free on Google Colab.
The skills transfer directly to Vertex AI / OpenAI fine-tuning later.

**Concepts**
- When to fine-tune: consistent style/format/behavior — *not* for adding knowledge (that's RAG)
- Dataset format: input → output pairs in JSONL; quality beats quantity
- LoRA / QLoRA: train a small adapter instead of the whole model, so it fits on a free GPU
- Train / validation split, loss curves, overfitting
- Evaluating base model vs fine-tuned model on the same test set

**What we'll build**
| File | Does |
|---|---|
| `make_dataset.py` | Uses Gemini (from Step 2) to generate + clean ~300 training examples for one narrow task, e.g. "turn casual notes into a formal ticket in our exact format" |
| `data/train.jsonl`, `data/val.jsonl` | The training data |
| `finetune_gemma.ipynb` | Colab notebook: load a small Gemma with **Unsloth**, attach LoRA, train, save the adapter |
| `compare.py` | Same prompts → base Gemma vs fine-tuned Gemma vs Gemini + few-shot prompt; score them |

**Needs:** a Google account for Colab (free T4 GPU), optionally a Hugging Face account to download Gemma.

**Exercises**
- Train with 50 vs 300 examples and compare.
- Honest question: did fine-tuning beat a good few-shot prompt from Step 2? (Often it doesn't — that's a key lesson.)

**Done when:** you have a fine-tuned adapter that follows your format more reliably than the base model, and you can explain whether it was worth it.

---

## Rough timeline

| Step | Effort |
|---|---|
| 2. Prompt Lab | 1–2 sessions |
| 3. Mini RAG | 2–3 sessions |
| 4. Tool Calling | 2 sessions |
| 5. Fine-Tuning | 2–3 sessions |

## Ground rules for every step
- Secrets stay in `.env`; new dependencies go in `requirements.txt`.
- Each folder gets a short `README.md` with how to run it and what it teaches.
- Free tier has rate limits — scripts keep the retry setup from Step 1 and use small datasets.
