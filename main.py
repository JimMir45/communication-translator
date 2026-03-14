"""
Communication Translator - FastAPI Backend

Endpoints:
  GET  /                  Serve frontend
  GET  /api/config        Get current LLM config (token masked)
  GET  /api/directions    List available translation directions
  POST /api/detect        Auto-detect input direction
  POST /api/translate     Streaming SSE translation
"""

import json
import os
from typing import AsyncGenerator

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from prompts import DETECT_SYSTEM, DIRECTIONS, build_translate_message

load_dotenv()

app = FastAPI(title="Communication Translator", version="1.0.0")

# ---------------------------------------------------------------------------
# Defaults from environment
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
DEFAULT_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    base_url: str = Field(default="", description="API base URL, empty = use server default")
    token: str = Field(default="", description="API token, empty = use server default")
    model: str = Field(default="", description="Model name, empty = use server default")


class TranslateRequest(BaseModel):
    direction: str = Field(..., description="pm_to_dev | dev_to_pm | to_ops")
    content: str = Field(..., min_length=1, max_length=8000)
    llm_config: LLMConfig = Field(default_factory=LLMConfig)


class DetectRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=8000)
    llm_config: LLMConfig = Field(default_factory=LLMConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_llm_config(req_cfg: LLMConfig) -> tuple[str, str, str]:
    """Merge request-level config with server defaults. Request takes priority."""
    base_url = req_cfg.base_url.strip() or DEFAULT_BASE_URL
    token = req_cfg.token.strip() or DEFAULT_TOKEN
    model = req_cfg.model.strip() or DEFAULT_MODEL
    return base_url, token, model


def make_client(base_url: str, token: str) -> anthropic.Anthropic:
    if not token:
        raise HTTPException(status_code=400, detail="API token is not configured.")
    return anthropic.Anthropic(api_key=token, base_url=base_url)


async def stream_anthropic(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    user_message: str,
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted chunks from Anthropic streaming API."""
    try:
        with client.messages.stream(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                # SSE format: data: <payload>\n\n
                payload = json.dumps({"type": "delta", "text": text}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except anthropic.AuthenticationError:
        payload = json.dumps({"type": "error", "message": "API 认证失败，请检查 Token 是否正确。"})
        yield f"data: {payload}\n\n"
    except anthropic.APIConnectionError:
        payload = json.dumps({"type": "error", "message": "无法连接到 API，请检查 Base URL 是否正确。"})
        yield f"data: {payload}\n\n"
    except anthropic.RateLimitError:
        payload = json.dumps({"type": "error", "message": "请求频率超限，请稍后重试。"})
        yield f"data: {payload}\n\n"
    except anthropic.BadRequestError as e:
        payload = json.dumps({"type": "error", "message": f"请求参数错误：{e.message}"})
        yield f"data: {payload}\n\n"
    except Exception as e:
        payload = json.dumps({"type": "error", "message": f"翻译失败：{str(e)}"})
        yield f"data: {payload}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/config")
def get_config():
    """Return current server-side LLM config. Token is masked for security."""
    token = DEFAULT_TOKEN
    masked = f"{token[:8]}...{token[-4:]}" if len(token) > 12 else ("***" if token else "")
    return {
        "base_url": DEFAULT_BASE_URL,
        "token_masked": masked,
        "model": DEFAULT_MODEL,
    }


@app.get("/api/directions")
def get_directions():
    """Return available translation directions."""
    return [
        {"value": key, "label": cfg["label"]}
        for key, cfg in DIRECTIONS.items()
    ]


@app.post("/api/detect")
def detect_direction(req: DetectRequest):
    """
    Auto-detect which translation direction suits the input.
    Returns direction, confidence, and reason.
    """
    base_url, token, model = resolve_llm_config(req.llm_config)
    client = make_client(base_url, token)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=256,
            system=DETECT_SYSTEM,
            messages=[{"role": "user", "content": req.content}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        return result
    except json.JSONDecodeError:
        # Fallback: return pm_to_dev as default
        return {"direction": "pm_to_dev", "confidence": "low", "reason": "无法解析，默认返回产品→开发"}
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="API 认证失败，请检查 Token。")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=502, detail="无法连接到 API，请检查 Base URL。")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/translate")
def translate(req: TranslateRequest):
    """
    Streaming translation endpoint.
    Returns Server-Sent Events (SSE) stream.
    """
    if req.direction not in DIRECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"无效的翻译方向: {req.direction!r}，可选: {list(DIRECTIONS)}"
        )
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="输入内容不能为空。")

    base_url, token, model = resolve_llm_config(req.llm_config)
    client = make_client(base_url, token)
    system, user_message = build_translate_message(req.direction, req.content)

    return StreamingResponse(
        stream_anthropic(client, model, system, user_message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )


# ---------------------------------------------------------------------------
# Static files (frontend) — mounted last so API routes take priority
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
