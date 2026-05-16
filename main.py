"""
SHL Assessment Recommender – FastAPI service
--------------------------------------------
GET  /health  →  {"status": "ok"}
POST /chat    →  {"reply": ..., "recommendations": [...], "end_of_conversation": ...}

Schema is non-negotiable per the assignment spec. Any deviation breaks the
automated evaluator.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv()  # load .env before anything reads os.environ

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lazy imports (heavy; loaded once at startup) ──────────────────────────────

_agent = None


def _ensure_catalog() -> None:
    """Download catalog.json from the official URL if it is missing."""
    if Path("catalog.json").exists():
        return
    logger.info("catalog.json not found — downloading from official URL…")
    try:
        import requests  # type: ignore
        from download_catalog import CATALOG_URL, normalise

        r = requests.get(CATALOG_URL, timeout=60)
        r.raise_for_status()
        try:
            raw = r.json()
        except Exception:
            import json as _json
            raw = _json.loads(r.text, strict=False)
        import json

        items = [normalise(item) for item in raw if item.get("name") and item.get("link")]
        Path("catalog.json").write_text(
            json.dumps(items, indent=2, ensure_ascii=False)
        )
        logger.info("Downloaded %d assessments → catalog.json", len(items))
    except Exception as exc:
        logger.error("Could not download catalog: %s", exc)
        raise RuntimeError(
            "catalog.json missing and could not be downloaded. "
            "Run: python download_catalog.py"
        ) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    t0 = time.time()
    logger.info("Starting up…")
    _ensure_catalog()
    from agent import SHLAgent
    _agent = SHLAgent()
    logger.info("Agent ready in %.1fs", time.time() - t0)
    yield
    logger.info("Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Pydantic models (strict per assignment schema) ────────────────────────────

class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def check_role(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v

    @field_validator("content")
    @classmethod
    def check_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("content must not be empty")
        return v


class ChatRequest(BaseModel):
    messages: List[Message]

    @field_validator("messages")
    @classmethod
    def check_messages(cls, msgs: List[Message]) -> List[Message]:
        if not msgs:
            raise ValueError("messages must not be empty")
        # Evaluator caps at 8; silently trim rather than reject
        # 8 turns × 2 messages/turn = 16 max; trim to preserve full context
        return msgs[-16:]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Readiness probe. Returns 200 as soon as the agent is loaded."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready yet")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        result = _agent.process(messages)
    except Exception as exc:
        logger.exception("Agent error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        reply=result["reply"],
        recommendations=[
            Recommendation(
                name=r["name"],
                url=r["url"],
                test_type=r.get("test_type", ""),
            )
            for r in result.get("recommendations", [])
        ],
        end_of_conversation=bool(result.get("end_of_conversation", False)),
    )


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def _global(request: Request, exc: Exception):
    logger.error("Unhandled: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
