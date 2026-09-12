# OrderOrder — Tech Stack

| | |
|---|---|
| **Version** | 0.3 (built; the plan and what was built, side by side) |
| **Date** | 9 September 2026 (0.2 written 4 September) |
| **Companion documents** | [PRD.md](PRD.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [ROADMAP.md](ROADMAP.md) · [DEPLOYMENT.md](DEPLOYMENT.md) |

Constraints this stack satisfies: **₹0 for the hackathon**, **open and official data only**, **permissive licences only in the product path**, a **CPU-only development machine**, **LangChain + LangGraph** as the orchestration layer, **self-hosting as the production target** for privileged documents, and a team of two.

The version 0.1 stack assumed a rented GPU for the demo and Pydantic AI for orchestration. Both are gone. Every provider, model and service below is either free of charge within a published allowance or runs on the development machine; the paid and self-hosted pieces are kept only as the production profile, reachable by changing environment variables.

**What version 0.3 adds.** The thing is built, and five of the choices below did not survive contact
with it. The document parser, the OCR stack, the front-end framework, the database and the embedding
model are all something other than what §2 specified — in three of those cases because the specified
component turned out to be solving a problem this corpus does not have, and in one because it was
measured and made retrieval *worse*. Each is marked in place with what replaced it and why, rather
than quietly amended, because the reason a choice was abandoned is more useful than the choice was.
The research in §3, §5 and §6 — licences, free-tier terms, GPU allowances — is unchanged and is still
what the decisions rest on. `uv.lock` is the authority on what is actually installed.

---

## 1. Principles

1. **Free first, self-hosted later.** The hackathon build spends nothing. The model behind every call is a configuration string, so moving from a free API tier to a self-hosted GPU box is not a code change.
2. **Demo data only on free tiers.** Several free tiers train on what you send them (§5.1). The hackathon processes moot memorials and synthetic matters, never client documents. The one free provider that does not train on inputs is reserved for anything sensitive; production keeps privileged text on the team's own hardware.
3. **Boring infrastructure.** One database (PostgreSQL with pgvector, which also stores LangGraph checkpoints), no queue for the hackathon, Docker only for Postgres. Add components when a measurement demands them.
4. **Deterministic pipeline, model at the leaves.** The verification engine is a LangGraph state graph whose nodes are typed Python functions. Language-model calls sit inside specific nodes with a Pydantic output schema. No node runs a ReAct-style agent loop.
5. **Permissive licences.** MIT, Apache-2.0, BSD and CC-BY only in the product path (§3).
6. **Same code, three profiles.** Local-offline (Ollama, small model), hackathon-free (free API tiers with fallbacks, free GPU notebooks for batch work), production-self-hosted (SGLang, TEI, PaddleOCR-VL on a rented GPU box).

---

## 2. Stack at a glance

Three columns, because the middle one is the interesting one. **Specified** is what version 0.2 chose
on 4 September; **built** is what is in `uv.lock` and the source today; where they differ the reason
is in the last column and, at more length, in the section referenced.

| Layer | Specified (0.2) | Built | Why they differ |
|---|---|---|---|
| Orchestration | **LangChain 1.4 + LangGraph 1.2**: `StateGraph` for the engine, `init_chat_model` + `with_fallbacks` for providers, `with_structured_output` for typed results | As specified. `engine/graph.py` is the compiled graph and every entry point goes through it | — |
| Agent layer | **Strands Agents SDK 1.55**: one `Agent` over eight `@tool` functions, one per check, in `agent/tools.py`. Bedrock as the model provider, falling back to the provider chain above through LiteLLM when no AWS credential resolves | Built. `orderorder agent` and `POST /api/agent`. The agent chooses which check runs; it has no path to the knowledge base except through a tool, and the verification graph is untouched (ARCHITECTURE §15) | — |
| LLM | Free API tiers behind ordered fallbacks; Ollama offline | As specified. `engine/providers.py`, chain from `LLM_PRIMARY` and `LLM_FALLBACKS` | The primary model *id* changed: see §5.1 |
| Frontend | Next.js + TypeScript + Tailwind + shadcn/ui, react-pdf | **One static HTML document** with its own CSS and JS and self-hosted fonts, served by the same FastAPI process. No build step, no Node, no second runtime | The page is a verdict board, a judgment viewer, a search box and a drafting workspace. A toolchain to deploy alongside the engine bought none of that, and every dependency it added was one more thing between a lawyer and the corpus |
| API | Python 3.12 via `uv`, FastAPI, Pydantic v2, SQLAlchemy 2, Alembic; background tasks with a jobs table | As specified, except jobs: an **in-process registry with a fifteen-minute deadline**, not a table. The `job` table is declared and unused | A job is worthless once the tab closes. [DEPLOYMENT.md](DEPLOYMENT.md) §2.3 — it means **run exactly one worker** |
| Parsing | Docling (MIT) through `langchain-docling`; pypdfium2 for text-layer detection | **pypdfium2 alone**, plus a cleaner written against this publisher (`ingest/pdf.py`) | The SCR PDFs are born-digital and uniform, so the problem was never layout. It was telling the reporter's words from the court's. §7 |
| OCR | PP-OCRv6 on CPU; PaddleOCR-VL on a Kaggle GPU | **Not built.** A brief with no text layer is reported as such | Nothing in the corpus needs it. It arrives when a scanned brief does. §7 |
| Embeddings | BGE-M3 on Kaggle, queries on local CPU | **A static encoder (model2vec)**, four minutes over the corpus — built, measured, and **off by default because it lowers recall at this scale** | The measurement is [ARCHITECTURE.md](ARCHITECTURE.md) §11.4 and it is the most useful number in this project. §8 |
| Reranker | bge-reranker-v2-m3 on CPU | **Not built.** `RERANKER_MODEL` is configured and nothing reads it | Reranking a field the encoder cannot rank is not the missing piece; §11.4 says which is |
| Vector + full-text | PostgreSQL 17 + pgvector 0.8 in Docker, hybrid search in SQL with RRF | **SQLite + FTS5** by default; dense half a memmapped float16 matrix beside it. **Chroma** is an optional second home for the same vectors, behind the `chroma` extra. Postgres + pgvector wired in compose and optional | 668,272 rows is a matrix multiply. A service to do that would have been a service to run — but a database earns its keep when something *outside* this process wants to query. §8 |
| Rhetorical roles | A local LLM now, a fine-tuned InLegalBERT later | **A cue classifier**, `ingest/roles.py`, abstaining to `none` where nothing matches | A cue is a phrase in the judgment, so a disagreement is settleable by looking. A model's label is not |
| Checkpoints | `langgraph-checkpoint-postgres` | Both checkpointers installed; SQLite is the default path | Follows the database |
| Tracing and eval | LangSmith Developer plan; DeepEval in CI | **One stderr logger**, `ORDERORDER_LOG_LEVEL`, prompts never logged. The eval harness is `orderorder eval`, six subcommands, reports checked into `evals/` | The harness needed to run with no account and no network. What tracing would have caught, one log line did — see [ARCHITECTURE.md](ARCHITECTURE.md) §12 |
| Auth | Auth.js | **One bearer token** the operator generates; the binding decides whether it is required | No accounts, so nothing to log in to. [DEPLOYMENT.md](DEPLOYMENT.md) §3a |
| Storage | Local disk under `ORDERORDER_DATA_DIR` | As specified. Uploaded briefs are never stored at all | Stronger than a retention policy |
| Hosting | The development machine; Vercel/Render if a link is needed | localhost, or the `serve` compose profile in a container published to `127.0.0.1:8000` | Free hosts spin down and the corpus is 1.2 GB |
| Production profile | SGLang, TEI, PaddleOCR-VL, Postgres, MinIO, Langfuse, Caddy on a self-hosted GPU box | Unchanged as the target; none of it built | Privileged documents must not leave hardware the team controls |

## 3. Licence audit

| Component | Licence | Verdict |
|---|---|---|
| LangChain, LangGraph, langgraph-checkpoint-postgres, langgraph-checkpoint-sqlite, langchain-groq, langchain-google-genai, langchain-cerebras, langchain-ollama, langchain-openai | MIT | Use — **installed** |
| **strands-agents** (the agent layer), boto3/botocore (its Bedrock provider) | Apache-2.0 | Use — **installed** |
| **LiteLLM** (the agent's non-Bedrock fallback, pulled by `strands-agents[litellm]`) | MIT | Use — **installed** |
| **OrderOrder itself** | Apache-2.0 ([LICENSE](../LICENSE)) | The repository was unlicensed and `pyproject.toml` said `Proprietary`, which is not a licence and left everyone who could read the public repository with no right to do anything with it. Apache-2.0 rather than MIT for the express patent grant and the contribution terms, both of which matter more to a company built on this than the extra paragraphs cost |
| FastAPI, Pydantic, SQLAlchemy, Alembic, Typer, rich, rapidfuzz, pypdfium2, python-docx, model2vec, sentence-transformers, uvicorn, httpx, tenacity | MIT / BSD / Apache-2.0 | Use — **installed** |
| langchain-postgres, langchain-docling, langchain-huggingface, Docling, Next.js | MIT | Cleared, and **not installed**: the pieces they served are built otherwise (§2, §7, §8) |
| PaddleOCR, PaddleOCR-VL weights, pgvector, TEI, Qwen3.5, Qwen3-Embedding, bge models, gpt-oss, OpenNyAI code and NER | Apache-2.0 (pgvector: PostgreSQL licence) | Use |
| AWS Open Data SC and HC judgment datasets | CC-BY-4.0 | Use with attribution |
| OpenNyAI rhetorical-role data | CC-BY-SA-4.0 | Use for training; share-alike applies to derived datasets; confirm before redistributing labels |
| Indian Kanoon API data | Contract: mandatory "powered by IKanoon" attribution including RAG and fine-tuning; silent on caching | Lookup only until caching is confirmed in writing |
| Gemma 4 | Reported open licence permitting commercial use | Re-verify before shipping on it |
| PyMuPDF / pymupdf4llm, MinerU | AGPL-3.0 | **Exclude** (or buy a commercial licence) |
| Marker, Surya weights | OpenRAIL-M with a $5M revenue/funding cap | **Exclude** |
| Jina reranker weights (self-hosted) | CC-BY-NC-4.0 | **Exclude** as weights; the hosted API's free tokens are usable (§8) |
| IL-TUR | CC-BY-NC-SA-4.0 | **Exclude** |
| Llama 4 | Llama Community Licence | Exclude; better Apache-2.0 options exist |
| Kimi K3 | Modified MIT with a revenue trigger | Exclude |
| Vercel Hobby | Terms prohibit commercial use | Hackathon only |

---

## 4. LangChain, reconsidered

### 4.1 What changed and why

Version 0.1 chose Pydantic AI and plain Python because the engine must be deterministic and testable stage by stage. That requirement stands. What changed is the budget: a ₹0 build lives on several free API tiers with different rate limits, context caps and data terms, and it must fail over between them mid-run. That is LangChain's core competence, and LangGraph turns out to be the natural implementation of the engine that ARCHITECTURE.md already specified:

- **Provider swapping is a string.** `init_chat_model("google_genai:gemini-3.6-flash")`, `init_chat_model("groq:openai/gpt-oss-120b")`, `init_chat_model("cerebras:gpt-oss-120b")`, `init_chat_model("ollama:qwen3.5:4b")` share one interface; `.with_fallbacks([...])` chains them so a 429 from one provider silently moves to the next.
- **The engine is already a state machine.** ARCHITECTURE.md Diagram 3 (the stage flow) and Diagram 8 (the verdict state machine) map one-to-one onto a LangGraph `StateGraph`: nodes are stages, conditional edges are decisions, the state is the verdict-in-progress. A checkpointer makes every run resumable and every intermediate state inspectable. `interrupt()` implements `needs_review` as a real pause for a human decision, resumed with `Command(resume=...)`. The node map is in ARCHITECTURE.md §4.13.
- **Typed outputs survive the switch.** `with_structured_output(PydanticSchema)` gives the same Pydantic objects the 0.1 design relied on, with the method chosen per provider (§4.2).
- **Integrations we would otherwise write.** This was the weakest of the five reasons and it is the one that did not pay: `langchain-docling`, `langchain-huggingface` and `langchain-postgres` were each the answer to a question that turned out to have a different answer (§7, §8), and none is installed. The four that did pay — provider swapping, the state machine, typed outputs, legibility — were enough on their own.
- **Legibility.** A LangGraph graph is something judges and new contributors can read.

### 4.2 How it is used

| Concern | Choice |
|---|---|
| Graph | One `StateGraph` per surface: `verify_citation` (invoked once per citation, fanned out over a brief with `Send`) and `draft_matter`. State is a Pydantic model: the §8 verdict object plus working fields |
| Nodes | Plain Python functions in `engine/graph.py`, each delegating to one module under `engine/`; each is unit-tested with a fake model. The resolver, quote verifier, voice, truncation, hierarchy, citator and contrary search never call a model at all |
| Models | `init_chat_model` with provider strings from environment variables; a `providers.py` module builds the primary model and its fallback chain (§5.2) |
| Structured output | `with_structured_output(schema, method=...)`: `json_schema` for Gemini and Ollama (native schema support), `function_calling` for Groq, Cerebras and Mistral (the safest method on open-weight endpoints), never `json_mode` |
| Prompts | `engine/prompts.py`; the version string is stamped into every verdict |
| Retrieval | Plain functions in `engine/lexical.py` and `engine/search.py`, not a `BaseRetriever`: nothing in the engine invokes retrieval through LangChain, so the abstraction had no caller. The reasoning that ruled out a stock vector store still holds — it does no reciprocal-rank fusion and no exact citation lookup |
| Checkpointer | Both are installed and follow `DATABASE_URL`; the default path is SQLite. The graph is compiled without one for a single synchronous verification, which is what every entry point does today |
| Human in the loop | Designed, not built. `needs_review` is returned and shown as a first-class state, but nothing pauses on it — there is no reviewer to resume it until there are accounts ([ARCHITECTURE.md](ARCHITECTURE.md) §13) |
| Tracing | `LANGSMITH_TRACING=true` plus an API key during the hackathon; the Langfuse callback handler in production |
| Evaluation | `orderorder eval`, six subcommands, is the source of truth; reports are checked into `evals/` so a number can be read back to the run that produced it. Neither the LangSmith dataset mirror nor DeepEval was built — the harness had to run with no account and no network, which is also what lets the suite exercise it with a stub model |

### 4.3 What is not used

- `create_agent` and any tool-calling agent loop **in the verification path**. LangGraph is used as a state machine, not as an agent runtime, and that has not changed: the agent layer added later (§2, ARCHITECTURE §15) sits *above* the graph and calls it as one tool among eight, so the path a citation takes through the engine is still fixed and still inspectable. Where a tool-calling loop does run, it runs on Strands rather than on LangGraph, because nothing in the engine needed to become an agent for a question to be routed to it.
- The legacy chains that moved to `langchain-classic` in 1.0 (`LLMChain`, `SequentialChain`, `ConversationalRetrievalChain`, `AgentExecutor`).
- LangGraph Platform. The open-source checkpointers are all that is needed, and they are free.

### 4.4 Version pins (PyPI, 4 September 2026)

| Package | Version | Note |
|---|---|---|
| `langchain` | 1.4.0 | 1.0 was GA on 22 Oct 2025 and is a long-term-support line; Python 3.10+ |
| `langchain-core` | 1.6.1 | |
| `langgraph` | 1.2.11 | |
| `langgraph-checkpoint-postgres` | current | MIT |
| `langchain-google-genai` | 4.4.0 | |
| `langchain-groq` | 1.1.x | |
| `langchain-cerebras` | 0.8.2 | Last release Nov 2025; pin and watch |
| `langchain-ollama` | current | |
| `langchain-huggingface` | 1.2.2 | |
| `langchain-postgres` | 0.0.17 | Use `PGVectorStore`, not the deprecated `PGVector` |
| `langchain-postgres`, `langchain-docling`, `langchain-huggingface` | — | Cleared and not installed; see §4.2 |

Sources: [1.0 announcement](https://blog.langchain.com/langchain-langgraph-1dot0/) · [release policy](https://docs.langchain.com/oss/python/release-policy) · [models and `init_chat_model`](https://docs.langchain.com/oss/python/langchain/models) · [persistence and checkpointers](https://docs.langchain.com/oss/python/langgraph/persistence) · [v1 migration guide](https://docs.langchain.com/oss/python/migrate/langchain-v1).

---

## 5. Language-model layer

### 5.1 Free API tiers (checked 4 September 2026; limits change, so re-check the linked pages before the demo)

> **A model id is not a constant, and a retired one fails silently.** `gemini-2.5-flash` was the
> configured primary in this table and is retired: Google answers 404 for it on new keys and names
> `gemini-3.6-flash` as the replacement. It failed in the way that costs the most. Every model call in
> this engine degrades to *not assessed* by design, so a dead model is indistinguishable from a corpus
> with nothing to say — one held-out evaluation scored 100% abstention and mode 4 at 0/14 and **read
> like a result**. It was found by adding a log line, not by the run looking wrong.
> `orderorder doctor --probe` makes one real call and names the reason in a sentence; run it before an
> evaluation and again whenever a report comes back emptier than the last one. The defaults below are
> now `gemini-3.6-flash`; treat every other id here as re-checkable rather than settled.


| Provider | Free models | Published limits | Context | Structured output | Data terms | Role |
|---|---|---|---|---|---|---|
| **Google AI Studio** | Gemini 3.6 Flash (2.5 retired, see above); Flash-Lite | Flash: 10 RPM, 250K TPM, 500-1,500 RPD (sources conflict; check the [rate-limit dashboard](https://aistudio.google.com/rate-limit)); Flash-Lite: 15 RPM, 1,500 RPD | 1M tokens | Native JSON schema | **Free-tier prompts are used to improve Google products** ([pricing](https://ai.google.dev/gemini-api/docs/pricing)) | **Bulk and demo workhorse**: whole-judgment digests in one call; per-claim verification during the live demo. Demo data only |
| **Groq** | `openai/gpt-oss-120b`, Llama 3.3 70B, gpt-oss-20b | 30 RPM, 1,000 RPD, **8K TPM, 200K TPD** ([rate limits](https://console.groq.com/docs/rate-limits)); the console labels this the Developer plan and adding a card raises limits, which is not a ₹0 option | 131K | JSON schema and tool calling | States it does not train on inputs or outputs and offers zero data retention; confirm in the console terms | **Privacy-safe path** for anything resembling real text; too little token throughput to carry a whole brief alone |
| **Cerebras** | `gpt-oss-120b`, `zai-glm-4.7` | 5-15 RPM (sources differ), 30K TPM, **1M tokens/day**; an 8,192-token context cap on the free tier is reported ([free endpoint notes](https://pricepertoken.com/endpoints/cerebras/free)) | 8K (free) | Tool calling | No-training policy claimed; one source says the larger "Experiment" allowance requires opting into training; verify in the console | Fastest fallback for short verification prompts; unusable for digests |
| **Mistral La Plateforme** | Mistral Small, Medium, Magistral | "Experiment" tier, roughly 1B tokens/month at about 1 request/s; exact numbers are no longer published (see Admin Console › Limits) | 128K | Tool calling, JSON schema | **Trains on free-tier data by default** since 12 Mar 2026; opt out under Admin › Privacy ([data controls](https://docs.mistral.ai/admin/monitor-comply/privacy-data-controls)) | Large bulk allowance once opted out; second bulk provider for eval runs |
| Ollama, run locally | `qwen3:4b` (2.5 GB). `qwen3.5:4b` needs a newer ollama than 0.11.4, which refuses the pull with a 412 | Unlimited; **247 s per call measured** on four CPU cores with no usable GPU | 8-16K working | Native JSON schema | Local | Offline development, `doctor --probe`, last-resort fallback. Not a service: the model was not the problem — every answer came back as a filled schema and every quote verified — the memory was, and two thirds of each call was the model paging its own prompt back off disk |
| SambaNova | Llama 3.x, Llama 4 Maverick preview | 20 RPM, 200K tokens/day per model ([docs](https://docs.sambanova.ai/docs/en/models/rate-limits)) | varies | Tool calling | Free-tier data policy not documented | Optional extra fallback, demo data only |
| Not used | OpenRouter `:free` (50 requests/day and you must allow training and publishing of prompts), GitHub Models (8K input / 4K output caps), Hugging Face Inference Providers ($0.10/month credit), Cloudflare Workers AI (small models; possible embedding fallback only) | | | | | |

Anthropic's Claude API has no free tier in 2026 (new console accounts get a small one-time trial credit); it stays a paid, consented toggle for production (§5.5).

### 5.2 Routing policy

- **Primary for demo data and bulk:** Gemini Flash. It has 30 times Groq's token throughput and the only million-token context on a free tier, which is what whole-judgment digests need.
- **Fallback chain:** Gemini → Groq → Cerebras (prompts under 8K only) → Ollama. Built once in `providers.py` with `.with_fallbacks()` and `max_retries` for 429s.
- **Long-context calls** (digests, judgments over ~30K tokens): Gemini only, or a Kaggle batch job with a local model (§6).
- **Anything resembling real client text:** Groq only, or the production profile. Never Gemini, OpenRouter, SambaNova, or Mistral before opting out of training.
- **Two accounts, double the budget.** Each free tier is per account, so a second key usually raises a daily ceiling further than a fifth provider does. An entry names the variable holding its key with `#VARIABLE` — `google_genai:gemini-3.6-flash#GOOGLE_API_KEY_2` — because the provider SDKs each read one fixed variable, which is exactly the assumption a second account breaks. The variable is part of the entry's identity, so the same model on two accounts is two rungs of the chain rather than one deduped away. (Version 0.2 asserted this worked; it did not, and the syntax above is what makes it true.)
- **What the chain does not do.** `with_fallbacks` moves on when a call raises, and remembers nothing: an exhausted key costs one failed request per call for the rest of a run, with no cooldown and no circuit breaker. Longer chains also make the quiet failure *more* likely rather than less — every uncovered failure degrades to `not assessed`, so a chain exhausted end to end produces a page of honest abstentions indistinguishable from a corpus with nothing to say (§5.1). Watch the abstention rate; it is the signal that the chain is spent.
- **Structured output method per provider** as in §4.2. Every call still ends in the pure-Python quote verifier, so a weaker provider cannot make a claim "supported".

### 5.3 The demo's token budget

Digests are the expensive calls and they are precomputed on sprint day 9, so the live demo only pays for per-claim work.

| Step | Calls | Tokens (approx.) | Provider |
|---|---|---|---|
| Digests for the 8 demo judgments, precomputed | 8 | 400K | Gemini Flash (one call per judgment) |
| Stress-test: 8 citations × 5 calls (decompose, locate and quote, adjudicate, treatment memo, opposing-counsel memo) | 40 | 120K | Gemini primary, Groq and Cerebras as fallbacks |
| Draft mode: case digest, issues, ~6 propositions through the gate | 25 | 80K | Same |
| **Live total** | **65** | **200K** | Under Gemini's 250K TPM within the 7-minute run; under its RPD either way. Through Groq alone the same run would take about 25 minutes at 8K TPM, which is why Groq is the fallback, not the primary |

Sprint-day eval runs (50 gold items × 5 calls, about 750K tokens) fit inside a day on Gemini plus Mistral, and the once-per-judgment digest cache means the gold set's judgments are digested exactly once.

### 5.4 Local and self-hosted models

Development runs Ollama with Qwen3.5-4B at 4-bit for offline work and code-path testing; it is not used for verdict quality. The production profile serves open-weight models on a self-hosted GPU box, unchanged from version 0.1:

| Tier | Hardware | Model | Notes |
|---|---|---|---|
| 16-24 GB VRAM | RTX 4090/5090; cloud L4 or A10G | Qwen3.5-27B (Apache-2.0, 262K context) or Gemma 4 31B; gpt-oss-20b for the most reliable JSON | First production tier; ~30-60K working context after weights |
| 48-80 GB | A100, H100 | Qwen3.5-122B-A10B or gpt-oss-120b | Whole-judgment digests in one pass |

Serving: SGLang (prefix caching fits "one judgment, many claims"), vLLM as fallback, both OpenAI-compatible with JSON-schema constrained decoding, both Linux-only. Hugging Face TGI is archived and not used. Legal fine-tunes lose to 2026 general models on long-context reading; InLegalBERT is used only as a cheap tagger.

### 5.5 Paid cloud toggle (production, consented matters only)

Claude through the official `anthropic` SDK, model `claude-opus-5`; its citations feature returns verbatim `cited_text` with character offsets that feed the same quote verifier; Batches at half price for bulk digests. Enabled per matter with a consent record naming provider, model, retention setting and region, after confirming zero-data-retention eligibility for the chosen model.

---

## 6. Free GPU compute for batch work

A CPU-only machine cannot embed a million paragraphs or run PaddleOCR-VL. Free GPU notebooks can, as batch jobs whose outputs are imported into the local database.

| Service | Free allowance | Use |
|---|---|---|
| **Kaggle** | About 30 GPU-hours/week, 2×T4 (32 GB) or P100, 9-12 h sessions, internet on request, 20 GB persisted output ([Kaggle](https://www.kaggle.com/product-feedback/173129)) | Corpus embeddings with BGE-M3 (the hackathon subset in about an hour on 2×T4); digests with a local 9-14B model as an alternative to Gemini; PaddleOCR-VL over scanned annexures |
| **Modal** | $30/month compute credit, no card, no rollover (roughly 187 T4-hours) ([pricing](https://modal.com/pricing)) | Scheduled batch jobs such as a bi-monthly corpus refresh |
| **Lightning AI** | 15 credits/month, about 80 interruptible GPU-hours ([docs](https://lightning.ai/docs/team-management/academia/students)) | Spare capacity |
| Google Colab | T4, sessions not guaranteed, ~90-minute idle timeout | Backup only |
| Hugging Face ZeroGPU | 5 minutes/day on a free account | Not useful |

**Pattern.** A notebook reads the AWS Open Data subset directly from S3 (anonymous access), runs the model, writes parquet (paragraph id, embedding) or JSONL (digests) to its persisted output, and `orderorder ingest --embeddings <file>` imports it. Notebooks live in `notebooks/` and are versioned.

**Not for the live demo.** A Kaggle or Colab notebook can expose an OpenAI-compatible endpoint through a cloudflared or ngrok tunnel, but idle timeouts kill the kernel, the URL changes on restart, GPU allocation is not guaranteed, and both platforms' terms discourage serving. The live path is the free API tiers in §5.

---

## 7. Document parsing and OCR

**What was built: pypdfium2 and a cleaner, and no Docling.** Worth recording because the specified
component was not wrong about anything — it was solving a problem this corpus does not have.

Docling earns its place where a PDF's structure has to be recovered: mixed layouts, tables, columns,
reading order that the byte order does not give you. The official SCR PDFs in the open-data bucket are
born-digital and uniform, so the text layer comes out in reading order for free. The actual difficulty
is that the page carries **the reporter's words as well as the court's**, and a layout model has no
opinion about which is which. What had to be cut, and each is a rule in `ingest/pdf.py`:

- margin letters (the `A`-`H` column markers), running headers, the citation line, page numbers;
- the **editorial headnote** the Reports open with — the publisher's summary, stored separately and
  never used as the text a pinpoint resolves against;
- the **editors' sign-off** the Reports close with, which extraction ran together with the court's last
  paragraph in two thirds of the corpus. A quote verified against that would have been reported as the
  court's. Cutting it took 424,000 characters of publisher's text out of judgments already stored;
- the coram, read for bench strength — the metadata's judge column names only the *presiding* judge, so
  bench strength arrives under-counted from the parquet and is taken from the printed judgment instead.

None of that is layout analysis and all of it is publisher-specific. Docling returns the day a source
arrives whose problem actually is layout — High Court PDFs are the likely one.

**OCR is not built.** A brief filed as a scan is reported as having no text layer rather than guessed
at, which is the correct behaviour in the meantime: an OCR error in a quote is a false verdict, and
this engine's whole claim is that it does not produce those. The design stands for when a scanned
brief arrives, and the licence work behind it is still good:

- **Per-page routing** ([ARCHITECTURE.md](ARCHITECTURE.md) §3.2): text-layer pages take the path above,
  image-only pages go to OCR. Text-layer detection uses pypdfium2, not PyMuPDF.
- **PaddleOCR 3.7**: PP-OCRv6 on a local CPU for light scans; **PaddleOCR-VL-1.6** (0.9B parameters,
  109 languages including Hindi/Devanagari, Apache-2.0) on a Kaggle GPU for batches and as a production
  container. It wants 8 GB VRAM minimum and compute capability 8.0 for the vLLM-based path, which a T4
  lacks; use the standard PaddlePaddle inference path on Kaggle.
- **Tesseract** as a last resort only.
- **Evaluate before committing**: run PP-OCRv6 and PaddleOCR-VL over ~200 Indian judgment and annexure
  pages and pick per page type by measured character accuracy; vendor benchmarks do not include Indian
  court documents.
- Paid OCR APIs are not part of the hackathon build.

## 8. Retrieval and data layer

**Chunk = paragraph** with a judgment-context prefix (title, court, year, opinion type, role) prepended
before embedding.

### 8.1 What was built, and the measurement that decided it

The corpus is **one SQLite file** — 38,032 judgments (1950-2025), 707,647 paragraphs, of which 38,005
judgments hold text. Full text is an FTS5
table; ranking inside a judgment and across the corpus is BM25 with proximity. `DATABASE_URL` switches
to the Postgres design in §8.2, and the corpus must be re-ingested rather than copied.

The dense half exists and is **off by default**, which is the one part of this document that is a
result rather than a plan. `orderorder embed` gives every paragraph a vector and `engine/search.py`
fuses that ranking with the lexical one through reciprocal rank fusion. Fused at one vote it takes
paragraph recall on paraphrased queries **down** from 36% to 30%, and weighting it up makes that
monotonically worse — 27% at three votes, 25% at eight. `--dense` turns it on.

The reason is scale rather than fusion. Asked to pick the right paragraph out of a field of 400, a
static encoder gets it first 24 times in 40 and a small sentence transformer — 8.3 hours over this
corpus against 4 minutes — gets it 23. Neither discriminates at hundreds of thousands of candidates: a thousand times more
candidates is a thousand more chances to be nearer by accident. What that rules out is a night spent
on a bigger *CPU* model, which is worth knowing before spending it. What it leaves open is BGE-M3 on a
borrowed GPU, and that is now an experiment with a number to beat rather than an assumption. The store
records which model wrote the vectors and the encoder is a flag, so it is `orderorder embed --model
...` and a re-run. Full numbers: [ARCHITECTURE.md](ARCHITECTURE.md) §11.4.

**The store, given SQLite.** A memory-mapped float16 matrix beside the database and a list of paragraph
ids in the same order — 668,272 rows is a 340 MB matrix and a matrix multiply, and adding a vector
service to do that would have been adding a service. Float16 halves the file and costs nothing
measurable, since only the order of the scores matters. It is read in blocks, because holding all of it
resident is enough to get the process killed on a machine with no memory to spare.

Citation lookup is never vector search, in either design: normalised citation strings hit a unique
index on `citation_alias.normalized`.

**The reranker is not built.** `RERANKER_MODEL` is configured and nothing reads it. Reranking the top
20-40 does not help when the right paragraph is not in the top 400, which is what the paraphrase row
measures; the encoder is the piece to fix first.

**And the encoder can now be a good one.** `orderorder embed --api` encodes through any
OpenAI-compatible `/v1/embeddings` endpoint — `EMBEDDINGS_BASE_URL`, `EMBEDDINGS_API_KEY`,
`EMBEDDINGS_MODEL`. The 0.2 plan was BGE-M3 on a borrowed Kaggle GPU; a hosted endpoint is the same
model without the session, and the corpus is ~263-300M tokens, about **$3**. The static encoder stays
the zero-cost default and the measured baseline. This is also the seam the production profile's TEI box
plugs into, so the hosted endpoint is a stepping stone to self-hosting rather than a detour from it.

### 8.1a Chroma, and why it is an extra rather than a dependency

`engine/chroma_store.py`, with `orderorder chroma-import` and `orderorder chroma-search`. Ranking
still reads the memmap. What Chroma adds is a query surface for anything outside this process — a
dashboard, another service, an export, a metadata-filtered lookup a matrix cannot do. It is an import
rather than a re-embedding, so the backends cannot disagree; the query is encoded by whatever encoded
the corpus, read back from the store's index, because a model mismatch would rank silently wrong. It
embeds in-process, persists to a directory under `ORDERORDER_DATA_DIR`, and has telemetry off.

It sits behind `uv sync --extra chroma` for a security reason and not a packaging one. `chromadb`
1.5.9 is the newest release and carries **five open advisories with no fixed version published**
(PYSEC-2026-311, -3813, -3814, -3815). The CI audit runs against the locked *production* set, so
keeping chromadb out of that set is what lets the audit stay honest instead of being suppressed.

The consequence has to be said in the same breath: **enabling the extra takes on unpatched
vulnerabilities that CI will not warn about**, because the audit no longer sees the package. Verified:
`uv export --no-dev` omits chromadb and `pip-audit` reports the production set clean; the same export
`--extra chroma` reports the five. Nothing web-facing touches it — both commands are CLI only, so the
exposure is a local process reading a local directory, not a listening service.

**On-disk shape**, since it decides when this stops fitting:

| | size | share | |
|---|---|---|---|
| `paragraph` | 493 MB | 40% | the text itself |
| `paragraph_fts_content` | 468 MB | 38% | **a second copy of the same text** |
| `paragraph_fts_data` | 144 MB | 12% | the inverted index, the part that does the work |
| indexes on `paragraph` | 89 MB | 7% | |
| everything else | 28 MB | 2% | judgments, aliases, opinions, citations |

That is ~2.9 KB a paragraph, of which 1.1 KB is the duplicate: FTS5 as a standalone table keeps its own
copy of every body. An external-content table removes it, at the cost of a join and of rebuilding every
index. [DEPLOYMENT.md](DEPLOYMENT.md) §2.6 has the case for doing it before the corpus is large.

### 8.2 The Postgres profile (wired, optional, and what a larger corpus needs)

PostgreSQL 17 + pgvector 0.8 in Docker, volume under `ORDERORDER_DATA_DIR`; `halfvec` storage, HNSW
with `m = 16`, `ef_construction = 128`, `ef_search = 100`; `hnsw.iterative_scan` for filtered queries.
LangGraph checkpoints in the same database. Row-level security keyed on `matter_id` for user-scoped
tables, when there are any.

Hybrid search in SQL, scoped to one judgment for the locator and to the corpus for authority retrieval:

```sql
WITH dense AS (
  SELECT id, row_number() OVER (ORDER BY embedding <=> $1) AS r
  FROM paragraph
  WHERE text_version_id = $2
  ORDER BY embedding <=> $1
  LIMIT 40
),
lexical AS (
  SELECT id, row_number() OVER (ORDER BY ts_rank_cd(fts, q) DESC) AS r
  FROM paragraph, plainto_tsquery('english', $3) AS q
  WHERE text_version_id = $2 AND fts @@ q
  LIMIT 40
)
SELECT id, SUM(1.0 / (60 + r)) AS score
FROM (SELECT * FROM dense UNION ALL SELECT * FROM lexical) AS fused
GROUP BY id
ORDER BY score DESC
LIMIT 20;
```

| Corpus | Judgments | Paragraphs | Text | Vectors (halfvec, 1024-d) | Fits |
|---|---|---|---|---|---|
| Supreme Court 2013-2025 | 9,429 | 409,499 | 1.2 GB total, SQLite | not stored by default | one file |
| **Full Supreme Court 1950-2025, what is held today** | **38,032** (38,005 with text) | **707,647** | SQLite | 668,272 embedded, off by default | one file |
| Estimate this replaced | ~40-50k | ~3-5M | ~4-5 GB | ~6-10 GB (+ HNSW) | A 32-64 GB RAM box |
| High Courts (all 25) | ~17.8M | hundreds of millions | ~1 TB+ | ~1 TB | Phase 3: shard by court |

**Why not a free hosted database.** Neon's free plan is 0.5 GB and Supabase's is 500 MB (and pauses
after a week without requests); Qdrant Cloud's free cluster has 1 GB of RAM. None holds this corpus,
let alone its vectors — which is part of why the corpus is a file that lives on the box.

**The embedder bake-off**, still unspent and still the right way to choose: Voyage gives 200M free
tokens for the voyage-4 family and 50M for `voyage-law-2` ([pricing](https://docs.voyageai.com/docs/pricing));
Jina gives 10M shared tokens; Cohere's trial key allows 1,000 calls/month. Enough to compare embedders
on pinpoint hit@k, not enough to embed the corpus, and the corpus embedder decides the query embedder.

## 9. Developer machine setup

What the build assumes: an x86-64 machine with four cores or more, about 16 GB of RAM, and no NVIDIA GPU — the GPU work is deferred to free notebooks (§6) and, in production, to a rented box. Windows, macOS and Linux all serve; the Windows notes below are the ones that catch people out.

The constraint worth planning around is **disk**. Between the corpus, the Postgres volume, the model weights and the build caches this project wants tens of gigabytes, and every tool involved defaults to putting its share on the system drive. Choose a location with room, export it as `ORDERORDER_DATA_DIR`, and point the rest at it: `scripts/dev-env.sh` and `scripts/dev-env.ps1` do that for uv, Hugging Face and Ollama in one step. The two they cannot reach are Docker and a system-wide Ollama service, which are configured in their own settings.

Do these before the sprint (day 0). `$DATA` below is the directory chosen above.

1. **Check free space on the system drive.** A drive close to full will destabilise Windows updates and Docker, and the caches below are exactly what fills it.
2. **Move Docker's disk image off the system drive.** Docker Desktop → Settings → Resources → Advanced → Disk image location. On Windows, cap WSL memory in `%UserProfile%\.wslconfig` with `[wsl2]` and `memory=8GB`.
3. **Point Ollama's model store at `$DATA/ollama`.** Set `OLLAMA_MODELS` as a user environment variable, restart Ollama, then pull `qwen3:4b`. It has to be set *before* the server starts, or the pull lands on the system drive.
4. **Model caches under `$DATA`.** `HF_HOME` (BGE-M3, reranker, PaddleOCR models) and `UV_CACHE_DIR`.
5. **Python 3.12 via uv.** `winget install astral-sh.uv`, or the installer for your platform, then `uv python install 3.12`; `uv sync` creates the environment from `pyproject.toml`.
6. **Project data directory.** `ORDERORDER_DATA_DIR=$DATA` holds corpus files and the Postgres volume. Keep the code wherever you like, though a path outside a downloads folder is safer against cleanup tools.
7. **No Node.** The 0.2 stack wanted Node 22 and pnpm for a Next.js front end; the page is one static HTML document served by the API, so there is no JavaScript toolchain to install and nothing to build before `orderorder serve` works.
8. **Free accounts and keys** (both team members, one key each per provider): Google AI Studio (`GOOGLE_API_KEY`), Groq (`GROQ_API_KEY`), Cerebras (`CEREBRAS_API_KEY`), Mistral (`MISTRAL_API_KEY`, then opt out of training under Admin › Privacy), LangSmith (`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`), Kaggle (phone-verify to unlock GPU and internet), Hugging Face (model downloads), Indian Kanoon (`INDIANKANOON_TOKEN`, then apply for the non-commercial allowance). Optional: Voyage and Jina for the embedder bake-off. Verify each key with one call on day 0; keys live in `.env`, never in git.

What such a machine can and cannot do: run the whole application, the database, ingestion of the subset (minus embeddings), the engine end to end against free tiers or the local 4B model, the web UI and the evaluation harness. It cannot produce demo-quality verdicts locally or run PaddleOCR-VL; those go through the free tiers and Kaggle.

---

## 10. Application layer and Docker Compose

**API surface, as built.** Every route sits behind one middleware that checks the bearer token;
`/api/health` is the single open path.

| | |
|---|---|
| Verification | `POST /api/verify` (returns a job), `POST /api/upload` (PDF or DOCX, bounded at read time), `GET /api/jobs/{id}`, `GET /api/jobs/{id}/events` (server-sent), `GET /api/jobs/{id}/annotated`, `.../report`, `.../memo/{index}` |
| Drafting | `POST /api/plan`, `POST /api/draft` (returns a job), `GET /api/draft/{id}`, `.../events`, `.../document`, `.../document.docx` |
| Reading | `GET /api/judgment/{key}` (viewer payload), `GET /api/search` |
| The page | `GET /`, `/app.css`, `/app.js`, `/fonts/{name}` |

There are no `/matters` routes and no verdict override: both belong to the multi-tenant design that
has not been built, and [ARCHITECTURE.md](ARCHITECTURE.md) §13 says what arrives with it.

**Engine as a library.** `orderorder.engine.graph` exposes `verify_text` and `verify_citation` by
invoking the compiled LangGraph, with no web dependencies. The CLI is the primary surface and has
rather more in it than the API does: `verify`, `find`, `contrary`, `argue`, `draft`, `treatment`,
`locate`, `resolve`, `cite parse`, `ingest` (metadata, text, bulk-text, aliases, repair-trailers,
mark-opinions), `index`, `embed`, `citator`, `chroma-import`, `chroma-search`, `eval` (generate, run,
paraphrase, search, gate, contrary), `migrate`, `init-db`, `doctor`, `stats`, `serve`. `ingest` also
carries `mark-roles`, which labels every stored paragraph's rhetorical role by cue.

**Jobs** are an in-process registry, not a table: `web/jobs.py`, a fifteen-minute deadline checked
between citations, eviction that prefers finished jobs. **Run exactly one worker** —
[DEPLOYMENT.md](DEPLOYMENT.md) §2.3. arq on Redis arrives with the production profile.

**Migrations.** Alembic, with the environment inside the package at `src/orderorder/migrations` so it
ships in the wheel and is present in the container. `orderorder migrate` on an empty database;
`orderorder migrate --stamp` on one that already holds the corpus. `tests/test_migrations.py` asks
Alembic the same question `--autogenerate` asks, so a model that gains a column without a revision
fails the suite rather than the deployment.

**Exports**: python-docx for DOCX; Markdown for reports and the annotated brief.

**Compose**, as it is in `infra/docker-compose.yml` — three profiles, not one:

```yaml
services:
  postgres:  { image: pgvector/pgvector:pg17, profiles: [hackathon, prod, serve] }
  api:       { build: ., profiles: [serve], ports: ["127.0.0.1:8000:8000"] }   # FastAPI + engine + the page
  ollama:    { image: ollama/ollama, profiles: [offline] }
```

The `serve` profile declares `ORDERORDER_API_TOKEN` with **no default**, so a stack brought up without
one fails while compose is still interpolating. The corpus mounts at `/data` and stays out of the
image; keys arrive as environment and never enter a layer. The image runs as uid 10001, so a *bind*
mount needs `sudo install -d -o 10001 -g 10001 <dir>` before the first run. The production services —
`redis`, `worker`, `sglang`, `tei`, `paddleocr`, `minio`, `langfuse`, `caddy` — are the target
topology in [ARCHITECTURE.md](ARCHITECTURE.md) §10 and are not in the file yet.

Environment variables of note: `LLM_PRIMARY` (for example `google_genai:gemini-3.6-flash`),
`LLM_FALLBACKS` (comma-separated provider strings), `LLM_LONG_CONTEXT` (the provider allowed for
digests), `LLM_SENSITIVE` (the only provider allowed for non-demo text), `LLM_BASE_URL` (an
OpenAI-compatible gateway), `ORDERORDER_DATA_DIR`, `DATABASE_URL` (empty means SQLite),
`ORDERORDER_API_TOKEN`, `ORDERORDER_LOG_LEVEL`, `EMBEDDINGS_MODEL`, `RERANKER_MODEL` (configured,
unread), `INDIANKANOON_TOKEN`, `INDIANKANOON_DAILY_QUOTA`, `LANGSMITH_TRACING`. `.env.example`
documents every one of them and carries the warnings that cost time to learn.

---

## 11. Repository layout

```
lawbot/
  README.md                  what it does, what it scores, what is not built
  DESIGN.md                  the visual system for the page
  docs/                      PRD, architecture, tech stack, roadmap, deployment
  src/orderorder/
    cli.py                   every command; the primary surface
    config.py                Settings, read from environment and .env
    logs.py                  one stderr logger; prompts never reach it
    resolver.py              exact alias match, then fuzzy party names
    citations/               grammar for SCC, AIR, SCR, SCALE, JT, INSC, HC neutral, IK ids
    ingest/                  corpus client, metadata, pdf, segment, store, bulk, aliases, repair,
                             roles (rhetorical role per paragraph, by cue)
    engine/                  graph.py (the StateGraph), providers.py, and one module per check:
                             locator, quotes, scope, voice, weight, truncation, hierarchy, facts,
                             citator, search, contrary, lexical, embeddings, chroma_store,
                             sentences, memo, report, authority, verdict, prompts, schemas
    drafting/                plan, assemble, render, word, attack
    agent/                   tools.py (eight @tool wrappers over the checks), assistant.py (the
                             Strands agent and the prompt that forbids answering from memory),
                             model.py (Bedrock, or the provider chain through LiteLLM)
    evaluation/              generate, run, retrieval, gate, contrary, gold
    db/                      models.py, session.py
    migrations/              alembic env + versions; inside the package so it ships
    web/                     api.py, auth.py, limits.py, jobs.py, static/ (the page and its fonts)
  tests/                     48 modules; in-memory database, no network (test_chroma_store skips
                             unless the chroma extra is installed)
  evals/                     gold.jsonl, holdout.jsonl, paraphrases.jsonl, and the reports
  scripts/                   dev-env.sh, dev-env.ps1, gateway-failure-rate.py
  infra/docker-compose.yml
  demo/                      a brief, a plan, propositions, expected output, a scripted model
  Dockerfile                 non-root, uid 10001, corpus and keys outside the image
  data/                      gitignored; relocatable via ORDERORDER_DATA_DIR
```

There is no `apps/`, no `packages/` and no `notebooks/`: one installable package, one CLI, one page it
serves. The Kaggle notebooks in the 0.2 layout were for corpus embeddings, digests and OCR, none of
which is on the built path today.

---

## 12. Bill of materials

### 12.1 Hackathon: ₹0

| Item | Provider | Free allowance | The limit that binds | Terms |
|---|---|---|---|---|
| LLM, primary | Google AI Studio, Gemini Flash | 10 RPM, 250K TPM, 500-1,500 RPD, 1M context | RPM during a burst | Trains on free-tier data: demo data only |
| LLM, privacy-safe | Groq, gpt-oss-120b | 30 RPM, 1,000 RPD, 8K TPM, 200K TPD | TPM and TPD | No training; ZDR available |
| LLM, fallback | Cerebras, gpt-oss-120b | 5-15 RPM, 30K TPM, 1M tokens/day | 8K context | Verify training opt-in |
| LLM, bulk | Mistral Experiment tier | ~1B tokens/month, ~1 rps (unpublished) | RPS | Opt out of training first |
| LLM, offline | Ollama, `qwen3:4b`, run locally | Unlimited | 247 s per call on CPU, measured | Local |
| Corpus embeddings | A static encoder on a local CPU: ~4 minutes over 409,499 paragraphs when that was the corpus; 668,272 are embedded today. BGE-M3 through a hosted `/v1/embeddings` endpoint is now one flag (`embed --api`) and about **$3** for the corpus | free locally; ~$3 hosted | Not the bottleneck; §8.1 is | Local model, or any endpoint |
| Reranker | Not built | — | — | — |
| Embedder bake-off | Voyage (200M / 50M tokens), Jina (10M), Cohere trial (1,000 calls/month) | As listed | Tokens | API; public judgment text only |
| OCR | Not built; PP-OCRv6 and PaddleOCR-VL still the choice when a scanned brief arrives | Unlimited / 30 GPU-hours | — | Local |
| Scheduled batch | Modal | $30/month credit | Credit | — |
| Database and checkpoints | SQLite, one file (Postgres + pgvector in Docker, optional) | Local disk | Disk: ~3 KB a paragraph | Local |
| Tracing and eval datasets | Not used. `orderorder eval` runs offline; reports in `evals/`. LangSmith Developer (5,000 traces/month, 14-day retention) stays available | — | — | Cloud; demo data only |
| Frontend hosting | Not needed: the page is served by the API from the same process | — | — | — |
| API hosting, optional | Render free or Hugging Face Spaces | Spins down after 15 minutes / 48 hours | The corpus is 1.2 GB, which is the real obstacle to a public link | — |
| Case-law lookups | Indian Kanoon API | ₹500 signup credit; ₹10,000/month non-commercial allowance on approval; ₹0.20 per document after | Credits | "Powered by IKanoon" attribution |
| **Total** | | **₹0** | | |

### 12.2 Production (monthly, indicative, from the September 2026 research pass)

| Item | Cost |
|---|---|
| GPU box: E2E Networks A100-80G at ₹189/hour, business hours only | ~₹50,000 |
| Same, 24×7 | ~₹1.4 lakh (E2E spot at ~₹70/hour: ~₹50,000) |
| AWS Mumbai L4 (g6.xlarge) at ~$0.98/hour, 24×7 | ~$715 |
| CPU box for web, API, database if separate | ₹5,000-10,000 |
| Indian Kanoon pay-as-you-go | Under ₹5 per 30-citation brief |
| Storage, backups, domain, TLS | Under ₹3,000 |
| **Total** | **₹60,000 to 1.5 lakh depending on GPU choice** |

Per-unit economics are unchanged: a judgment digest costs ₹15-25 of GPU time once; each later verification of a claim against a digested judgment costs seconds of GPU time.

---

## 13. Security and privacy

Built:

- **The free-tier rule.** Free tiers that train on inputs (Gemini, Mistral before opt-out, OpenRouter, SambaNova) receive demo data only: moot memorials, synthetic matters, public judgment text. `LLM_SENSITIVE` names the single provider allowed for anything else during the hackathon (Groq), and the production profile keeps privileged text on the team's own hardware.
- API keys in `.env` files excluded from git; `.env.example` documents every variable; two sets of keys are never shared in chat. CI scans the **whole history** rather than the diff, since a key committed and later deleted is still published.
- Keys never enter an image layer: no `ARG`, no `COPY` of `.env`, `.dockerignore` excludes it. `docker history` would show one baked in even after a later layer deleted the file.
- **Uploaded briefs are never stored.** They are parsed in memory and handed straight back — not to a table, not to disk, and not to a log line. That is a stronger guarantee than encrypting them would be, and `tests/test_logs.py` is what holds the log half of it.
- One bearer token, required the moment the server is bound off loopback, or it refuses to start; rate limits on ordinary traffic and harder on failed authentication; security headers on every response. [DEPLOYMENT.md](DEPLOYMENT.md) §3a is the checklist item by item, including what does not apply and why.
- No user document is used for training or fine-tuning by the team; the free-tier rule above is what keeps that promise on the provider side.

Planned, and arriving with multi-tenancy, because none of it has an object to protect until matters belong to people:

- Postgres row-level security keyed on `matter_id`; integration tests asserting cross-matter reads fail.
- Uploads scanned before parsing; originals retained under the matter's prefix, if they are ever retained at all.
- An append-only audit log; delete-on-request removing the matter's rows, files, traces and cache entries.
- Encrypted volumes, TLS via Caddy, and worker containers with egress limited to the allow-listed judgment sources.

---

## 14. Deliberately not used

| Not used | Why |
|---|---|
| ReAct-style agents (`create_agent`) in the verification path | Verification must be deterministic and testable stage by stage; LangGraph is used as a state machine |
| The legacy chains now in `langchain-classic` | Superseded in LangChain 1.0 |
| LangGraph Platform | The open-source checkpointers are free and sufficient |
| Redis, arq, MinIO, Langfuse, Caddy in the hackathon profile | Nothing to gain for a two-person demo; all return in production |
| A rented GPU for the demo | The build must cost nothing; free API tiers serve the same open models |
| Tunnelled notebook endpoints for the live demo | Idle timeouts, changing URLs, unguaranteed GPUs, terms that discourage serving |
| OpenRouter free models, GitHub Models, Hugging Face Inference free credit, ZeroGPU | 50 requests/day with mandatory training consent; 8K/4K token caps; $0.10/month; 5 minutes/day |
| Free hosted Postgres for the corpus | 0.5 GB does not hold the vectors |
| Hugging Face TGI | Archived March 2026 |
| PyMuPDF, MinerU; Marker and Surya weights; Jina reranker weights; IL-TUR; Llama 4; Kimi K3 | Licences (§3) |
| Manupatra / SCC Online scraping | Prohibited by their terms |
| Model memory as a source of case law; sampling temperature as a safety control | Closed-world rule; grounding and string verification do that job |

Added after building it — things dropped for a measured reason rather than a principled one:

| Not used | Why |
|---|---|
| Docling, `langchain-docling` | The corpus is born-digital and uniform; the difficulty was the reporter's words, not the layout (§7) |
| A Next.js front end, and Node at all | One static page serves a verdict board, a viewer, a search box and a drafting workspace. The toolchain bought nothing and had to be deployed (§2) |
| pgvector as the default store | 668,272 rows is a 340 MB matrix and a matrix multiply. A vector service would have been a service to run — though Chroma is now available as an optional second home for queries from outside the process (§8.1a) |
| A reranker | It reorders the top 40. The paraphrase failure is that the right paragraph is not in the top 400 (§8.1) |
| Dense retrieval **on by default** | Built, fused, and measured: it lowers paragraph recall on paraphrases from 36% to 30%, worse the more weight it is given. `--dense` turns it on. This is the one entry here that is a result rather than a judgement ([ARCHITECTURE.md](ARCHITECTURE.md) §11.4) |
| A trained rhetorical-role classifier, and role labels at all | Voice and weight were built on the judgment's structure and its attributing cues instead, which yield a quotable reason rather than a label a reader cannot argue with |
| A model for treatment, voice, or the contrary search | Cue phrases, the citation graph and clause polarity do those, so no model call can invent a finding in any of them |

---

## 15. Sources

LangChain and LangGraph: [1.0 announcement](https://blog.langchain.com/langchain-langgraph-1dot0/) · [release policy](https://docs.langchain.com/oss/python/release-policy) · [models](https://docs.langchain.com/oss/python/langchain/models) · [persistence](https://docs.langchain.com/oss/python/langgraph/persistence) · [v1 migration](https://docs.langchain.com/oss/python/migrate/langchain-v1) · [LangSmith pricing FAQ](https://docs.langchain.com/langsmith/pricing-faq). Free LLM tiers: [Groq rate limits](https://console.groq.com/docs/rate-limits) · [Groq models](https://console.groq.com/docs/models) · [Gemini pricing and data terms](https://ai.google.dev/gemini-api/docs/pricing) · [AI Studio rate-limit dashboard](https://aistudio.google.com/rate-limit) · [Cerebras free endpoint](https://pricepertoken.com/endpoints/cerebras/free) · [Mistral data controls](https://docs.mistral.ai/admin/monitor-comply/privacy-data-controls) · [OpenRouter limits](https://openrouter.ai/docs/api-reference/limits) · [SambaNova rate limits](https://docs.sambanova.ai/docs/en/models/rate-limits) · [Cloudflare Workers AI pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/). Free compute: [Kaggle GPU quota](https://www.kaggle.com/product-feedback/173129) · [Modal pricing](https://modal.com/pricing) · [Lightning AI credits](https://lightning.ai/docs/team-management/academia/students) · [ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu). Embeddings and rerank: [Voyage pricing](https://docs.voyageai.com/docs/pricing) · [Jina embeddings](https://jina.ai/embeddings/) · [Cohere rate limits](https://docs.cohere.com/docs/rate-limits). Hosting and databases: [Vercel pricing](https://vercel.com/pricing) · [Render free tier](https://render.com/docs/free) · [Supabase pricing](https://supabase.com/pricing) · [Neon free plan](https://neon.com/faqs/free-plan-limits-and-quotas) · [Qdrant pricing](https://qdrant.tech/pricing/) · [Upstash pricing](https://upstash.com/pricing). Models, OCR, retrieval and data sources are unchanged from version 0.1: [PaddleOCR releases](https://github.com/PaddlePaddle/PaddleOCR/releases) · [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) · [Docling](https://github.com/docling-project/docling) · [pgvector](https://github.com/pgvector/pgvector/blob/master/CHANGELOG.md) · [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35) · [gpt-oss](https://openai.com/index/introducing-gpt-oss/) · [SGLang](https://github.com/sgl-project/sglang/releases) · [E2E Networks pricing](https://www.e2enetworks.com/blog/nvidia-a100-price-india) · [Indian Kanoon API](https://api.indiankanoon.org/documentation/) · [AWS Open Data SC](https://registry.opendata.aws/indian-supreme-court-judgments/).
