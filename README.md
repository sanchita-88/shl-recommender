# SHL Assessment Recommender

Conversational FastAPI agent that recommends SHL Individual Test Solutions through
dialogue — clarifying vague hiring intents, recommending grounded shortlists, refining
on constraint changes, and comparing assessments.

---

## Quick-start (local)

```bash
# 1. Clone / create the project folder
cd shl-recommender

# 2. Python environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Credentials
cp .env.example .env
# Edit .env → set GEMINI_API_KEY (free at https://aistudio.google.com/app/apikey)

# 4. Build the catalog  (choose ONE method)
#    Method A — requests-based (faster, try first):
pip install requests beautifulsoup4
python build_catalog_requests.py

#    Method B — Playwright (use if Method A gives an empty catalog.json):
pip install -r requirements-scraper.txt
playwright install chromium
python scraper.py

# 5. Run the service
uvicorn main:app --reload --port 8000
```

`GET http://localhost:8000/health` → `{"status":"ok"}`

---

## Example call

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a mid-level Java developer who works with stakeholders"}
    ]
  }'
```

Expected response shape:
```json
{
  "reply": "Great, for a mid-level Java developer with stakeholder interaction I'd suggest ...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r",       "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

---

## Run tests

```bash
# Make sure .env has a valid GEMINI_API_KEY and catalog.json exists
python test_agent.py -v
```

---

## Deploy to Render

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "initial"
gh repo create shl-recommender --public --push
```

> Make sure `catalog.json` is committed (not in `.gitignore`).  
> The catalog is static data, not secrets.

### Step 2 — Create service on Render

1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect your GitHub repo
3. Render auto-detects `render.yaml`; confirm:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Under **Environment**, add:
   - `GEMINI_API_KEY` = your key *(click "Add Secret Env Var")*
5. Click **Create Web Service**

Render will assign a URL like `https://shl-recommender.onrender.com`.

### Step 3 — Verify

```bash
curl https://shl-recommender.onrender.com/health
# → {"status":"ok"}
```

> **Free tier cold start:** Render free services sleep after 15 min.  
> The evaluator allows up to 2 minutes for `/health` to respond — this is enough.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | **Required** — Google AI Studio key |
| `LLM_PROVIDER` | `gemini` | `gemini` or `openai` (any OpenAI-compatible) |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `OPENAI_API_KEY` | — | Required if `LLM_PROVIDER=openai` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Override for Groq / OpenRouter |
| `OPENAI_MODEL` | `llama-3.3-70b-versatile` | Model for OpenAI-compatible providers |

### Using Groq instead of Gemini (alternative free tier)

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=gsk_...
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.3-70b-versatile
```

---

## Project structure

```
shl-recommender/
├── main.py                   # FastAPI app (GET /health, POST /chat)
├── agent.py                  # Agent logic: retrieval + LLM + schema enforcement
├── scraper.py                # Playwright-based catalog scraper
├── build_catalog_requests.py # requests+BS4 catalog scraper (faster, try first)
├── test_agent.py             # Offline test suite
├── catalog.json              # Pre-scraped SHL catalog (commit this!)
├── requirements.txt          # Production dependencies
├── requirements-scraper.txt  # Scraper-only dependencies
├── render.yaml               # Render deployment config
├── .env.example              # Environment variable template
└── APPROACH.md               # 2-page design document
```

---

## API specification

### `GET /health`
```
200 OK
{"status": "ok"}
```

### `POST /chat`

**Request**
```json
{
  "messages": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user",      "content": "..."}
  ]
}
```

**Response**
```json
{
  "reply": "string",
  "recommendations": [
    {"name": "string", "url": "string", "test_type": "string"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is empty `[]` while the agent is clarifying or refusing.
- `recommendations` has 1–10 items when the agent commits to a shortlist.
- `end_of_conversation` is `true` only when the task is complete.
- Max 8 turns per conversation; max 30 s per call.
