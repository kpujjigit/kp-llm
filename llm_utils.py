import os
import time
import hashlib
import torch
import sentry_sdk
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


def load_model():
    model_name = os.getenv("HF_MODEL_NAME", "gpt2")
    if os.path.exists("./trained_model"):
        model = GPT2LMHeadModel.from_pretrained("./trained_model")
    else:
        model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def generate_with_token_latency(model, tokenizer, prompt: str, max_new_tokens: int = 20):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    past_key_values = None
    output_ids = []
    token_latencies = []
    with torch.no_grad():
        for idx in range(max_new_tokens):
            with sentry_sdk.start_span(op="llm.token", description=f"token_{idx}") as span:
                start = time.perf_counter()
                outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                latency = (time.perf_counter() - start) * 1000
                span.set_tag("latency_ms", latency)
            token_latencies.append(latency)
            output_ids.append(next_token_id.item())
            input_ids = next_token_id
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return generated_text, token_latencies


def hash_prompt(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()
