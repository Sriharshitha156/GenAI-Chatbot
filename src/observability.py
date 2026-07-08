"""
Observability module — Day 5 Session 3
Exercise 1: logged_llm_call() wrapper — logs every LLM call with 7 fields.
"""
import os
import json
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_FILE     = os.path.join(_PROJECT_ROOT, "logs", "llm_calls.jsonl")

# In-memory session log — reset on every app restart
SESSION_LOGS: List[Dict[str, Any]] = []

# Cost per 1K tokens (USD) — GitHub Models gpt-4o-mini pricing
_COST_PER_1K = {
    "gpt-4o-mini":            {"input": 0.000150, "output": 0.000600},
    "gpt-4o":                 {"input": 0.002500, "output": 0.010000},
    "Phi-3.5-mini-instruct":  {"input": 0.000000, "output": 0.000000},  # free
}
_DEFAULT_COST = {"input": 0.000150, "output": 0.000600}  # fallback to gpt-4o-mini


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _COST_PER_1K.get(model, _DEFAULT_COST)
    return (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]


def _write_log(entry: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def logged_llm_call(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    purpose: str = "chat",       # e.g. "chat", "judge", "summarise", "ragas"
    **kwargs,
) -> Any:
    """
    Wrapper around client.chat.completions.create() that logs 7 fields:
      timestamp, model, input_tokens, output_tokens, latency, cost, status
    Returns the raw response object (or None on failure).
    """
    t0 = time.time()
    status = "success"
    response = None
    input_tokens = output_tokens = 0

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens  = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
        status = "success"
    except Exception as e:
        status = f"error: {type(e).__name__}: {str(e)[:120]}"

    latency = round(time.time() - t0, 3)
    cost    = _estimate_cost(model, input_tokens, output_tokens)

    entry = {
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose":       purpose,
        "model":         model,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "latency_s":     latency,
        "cost_usd":      round(cost, 6),
        "status":        status,
    }

    SESSION_LOGS.append(entry)
    _write_log(entry)

    if status != "success":
        return None
    return response
