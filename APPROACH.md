# SHL Assessment Recommender — Approach Document

## Problem Decomposition

The task is converting an open-ended, ambiguous hiring intent into a grounded shortlist
of SHL Individual Test Solutions — without recommending anything outside the catalog and
without stalling with unnecessary clarification questions.

Four distinct problems:

1. **Catalog ingestion** — normalise the provided JSON catalog into a structure the
   agent can search at runtime without losing any metadata.
2. **Retrieval** — given a free-form hiring description, surface the most relevant
   subset of assessments for the LLM to reason over.
3. **Agent logic** — decide when to clarify, recommend, refine, compare, or refuse, and
   do so within 8 turns.
4. **Schema enforcement** — guarantee every response matches the evaluator's fixed
   schema regardless of what the LLM returns.

---

## Catalog Ingestion

The provided catalog endpoint
(`tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json`)
delivers ~400 Individual Test Solutions in JSON. Each entry contains name, link,
description, job\_levels, languages, duration, remote/adaptive flags, and a `keys`
array of full category labels (`"Knowledge & Skills"`, `"Personality & Behavior"`,
etc.). A one-time normalisation step (`download_catalog.py`) converts labels to
letter codes (K, P, A, B, C, D, E, S), renames `link → url`, coerces the
yes/no strings to booleans, and writes `catalog.json`. The service auto-downloads
on first startup if the file is absent.

---

## Retrieval

**Embeddings.** Each assessment is represented as a text blob: name + type labels +
description + job levels. These are encoded with `sentence-transformers/all-MiniLM-L6-v2`
(384-dim, ~100 ms for the full corpus on CPU) and stored in a FAISS `IndexFlatIP`
(cosine on L2-normalised vectors).

**Query.** The last three user turns are concatenated as the retrieval query. This
naturally updates as the conversation evolves so refinements re-rank the candidates.
The top-20 candidates are injected verbatim into the LLM prompt.

This keeps the LLM context focused (~20 entries vs. 400) and eliminates hallucinated
URLs because the model can only pick names from what it is shown.

---

## Agent Design

**Single LLM call per turn.** The system prompt, full conversation history, and 20
catalog candidates are combined into one prompt. The model returns a JSON object with
four fields: `action`, `reply`, `selected_names`, `end_of_conversation`. The service
maps `selected_names` back to full catalog records (URL, test\_type). The LLM never
generates URLs — it only copies names.

**Decision rules (from sample conversations C1–C10):**

| Action | Trigger |
|--------|---------|
| `clarify` | Role entirely unknown; one targeted question |
| `recommend` | Role identified (even roughly); JD pasted; user refining |
| `compare` | "What is the difference between X and Y?" |
| `refuse` | Legal, salary, competitors, prompt injection |
| Force recommend | Turn ≥ 7 (injected as a system note) |

**Proactive behaviour** (learned from sample traces): OPQ32r is included for
personality and Verify G+ for cognitive on most senior/professional/graduate roles
unless the user declines. Catalog gaps (e.g. Rust, Kotlin) are acknowledged
explicitly; closest alternatives are offered. Language constraints on K-type tests
are flagged proactively.

**LLM.** Google Gemini 2.0 Flash (default; free tier, ~1–2 s latency, strong JSON
output). Switchable to any OpenAI-compatible provider via `LLM_PROVIDER` env var.

---

## Prompt Design

The system prompt is structured in six named sections (rules, test-type reference,
when to clarify, when to recommend, when to compare/refuse, output schema). Key
choices:

- **Flat prompt** (not multi-turn Gemini API): history + candidates concatenated into
  a single string avoids the Gemini SDK's history-role alternation constraint.
- **Candidates appended at the end**, close to the generation point, to reduce
  positional forgetting.
- **Turn counter** injected so the model can self-apply the force-recommend rule.
- **Temperature 0.1** for deterministic, evaluator-friendly JSON output.

---

## Evaluation Approach

**Hard evals:** `test_agent.py` checks schema compliance, SHL-only URLs, turn cap
(force-recommend at turn 7), and catalog-grounded recommendations on every run before
deployment.

**Behavior probes:** Tests cover all five actions (clarify, recommend, compare,
refuse, finalise), the refinement loop (add/drop/swap), gap handling (Rust → no
test), and the confirmation→EOC pattern from sample traces.

**Recall proxy:** Five archetype tests (tech, leadership, graduate, contact-centre,
safety) verify that the correct test-type *categories* surface without requiring
labeled ground truth.

**What didn't work:**
- Two-call design (intent extraction + generation) doubled latency and produced
  action/reply inconsistencies. Collapsed to one call.
- Dense retrieval alone sometimes missed exact assessment names. Fixed by appending
  the verbatim candidate list to the prompt so the model can name-match.
- Recommending on turn 1 for every query (eager) caused hallucinated URLs when
  candidates were irrelevant. Fixed by grounding strictly to the FAISS top-20.

---

## Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| API | FastAPI + uvicorn | async, typed, production-grade |
| LLM | Gemini 2.0 Flash | free tier, 1–2 s, strong JSON |
| Embeddings | all-MiniLM-L6-v2 | 384-dim, fast CPU inference |
| Vector search | FAISS IndexFlatIP | zero infra, in-process |
| Deployment | Render (free) | cold start ≤ 2 min, matches evaluator allowance |

**AI tools used.** Claude (Anthropic) assisted with code scaffolding and prompt
iteration. Architecture, retrieval design, system prompt logic, and test cases were
authored and verified manually against the sample conversation traces.
