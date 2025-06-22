import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

from sentry_setup import init_sentry
from llm_utils import load_model, generate_with_token_latency, hash_prompt

init_sentry()

app = FastAPI()
app.add_middleware(SentryAsgiMiddleware)

model, tokenizer = load_model()

class GenerationRequest(BaseModel):
    user_id: str
    prompt: str
    max_tokens: int = 20


@app.post("/generate")
async def generate(req: GenerationRequest):
    prompt_hash = hash_prompt(req.prompt)
    with sentry_sdk.start_span(op="llm.prompt_response", description="prompt+response"):
        start_time = time.perf_counter()
        text, token_latencies = generate_with_token_latency(model, tokenizer, req.prompt, req.max_tokens)
        generation_latency_ms = (time.perf_counter() - start_time) * 1000
    sentry_sdk.add_breadcrumb(
        category="llm",
        message="prompt processed",
        level="info",
        data={
            "user_id": req.user_id,
            "prompt_hash": prompt_hash,
            "token_count": len(token_latencies),
            "generation_latency_ms": generation_latency_ms,
        },
    )
    return {
        "text": text,
        "token_count": len(token_latencies),
        "latencies_ms": token_latencies,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
