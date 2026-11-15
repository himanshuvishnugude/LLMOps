import os, json, time, uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# 1. Replaced OpenAI import with Google GenAI
from google import genai

APP_NAME = "llm-app"
# 2. Updated default model to Gemini
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

LOG_FILE = Path.home() / "efk" / "logs" / "llm-app.jsonl"

# 3. Initialized the Gemini client (auto-detects GEMINI_API_KEY env var)
client = genai.Client()

app = FastAPI(title=APP_NAME)

def now():
    return datetime.now(timezone.utc).isoformat()

def write_log(event: dict):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/prompt")
async def prompt_endpoint(request: Request):
    request_id = str(uuid.uuid4())
    route = str(request.url.path)
    start = time.time()

    user_id = request.headers.get("x-user-id", "unknown")
    
    # 4. Uncommented tenant_id to prevent the NameError when write_log called it below
    tenant_id = request.headers.get("x-tenant-id", "unknown")

    body = await request.json()
    prompt = (body.get("prompt") or "").strip()

    write_log({
        "ts": now(),
        "event": "request_received",
        "app": APP_NAME,
        "route": route,
        "request_id": request_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "model": MODEL,
        "prompt_length": len(prompt),
    })

    llm_start = time.time()
    status = "success"
    error_type = None
    tokens_in = None
    tokens_out = None
    answer = None

    try:
        if not prompt:
            raise ValueError("empty_prompt")

        # 5. Replaced fake/OpenAI syntax with correct Gemini SDK generate_content method
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        # 6. Correctly extract the output text
        answer = resp.text

        # 7. Correctly extract the usage token counts using Gemini's usage_metadata
        if resp.usage_metadata:
            tokens_in = resp.usage_metadata.prompt_token_count
            tokens_out = resp.usage_metadata.candidates_token_count

    except Exception as e:
        status = "error"
        error_type = type(e).__name__
        answer = ""

    llm_latency_ms = int((time.time() - llm_start) * 1000)
    total_latency_ms = int((time.time() - start) * 1000)

    write_log({
        "ts": now(),
        "event": "response_sent",
        "app": APP_NAME,
        "route": route,
        "request_id": request_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "model": MODEL,
        "status": status,
        "llm_latency_ms": llm_latency_ms,
        "latency_ms": total_latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "error_type": error_type,
    })

    if status == "error":
        return JSONResponse(
            status_code=400,
            content={"request_id": request_id, "status": status, "error_type": error_type},
        )

    return {"request_id": request_id, "status": status, "response": answer}