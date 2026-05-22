from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from .settings import load_env_file


load_env_file()


def configured_model() -> str:
    model = os.getenv("CUET_LLM_MODEL", "").strip()
    if model:
        return model
    return configured_models()[0]


def configured_models() -> list[str]:
    base_url = configured_base_url()
    configured = [
        os.getenv("CUET_LLM_MODEL", "").strip(),
        *[item.strip() for item in os.getenv("CUET_LLM_FALLBACK_MODELS", "").split(",")],
    ]
    models = [item for item in configured if item]
    if models:
        return list(dict.fromkeys(models))
    if "hackclub.com" in base_url:
        return [
            "~openai/gpt-mini-latest",
            "~anthropic/claude-haiku-latest",
            "~google/gemini-flash-latest",
            "ibm-granite/granite-4.1-8b",
        ]
    return ["gpt-4o-mini"]


def configured_base_url() -> str:
    base_url = os.getenv("CUET_LLM_BASE_URL", "").strip()
    if base_url:
        return base_url.rstrip("/")
    return "https://ai.hackclub.com/proxy/v1" if os.getenv("HACKCLUB_AI_API_KEY", "").strip() else "https://api.openai.com/v1"


def configured_api_key() -> str:
    return (
        os.getenv("CUET_LLM_API_KEY", "").strip()
        or os.getenv("HACKCLUB_AI_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )


def chat_completion(
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    temperature: float = 0,
    timeout: int = 45,
) -> str:
    api_key = configured_api_key()
    if not api_key:
        raise RuntimeError("No LLM API key configured. Set HACKCLUB_AI_API_KEY, CUET_LLM_API_KEY, or OPENAI_API_KEY.")
    session = requests.Session()
    session.trust_env = False
    last_error: Exception | None = None
    for model in configured_models():
        try:
            response = session.post(
                f"{configured_base_url()}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens or int(os.getenv("CUET_LLM_MAX_TOKENS", "700")),
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"])
        except requests.RequestException as exc:
            last_error = exc
            if getattr(exc, "response", None) is not None and exc.response is not None and exc.response.status_code in {401, 403}:
                break
    raise RuntimeError(f"LLM request failed for configured model fallbacks: {last_error}")


def parse_json_object(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def parse_json_array(content: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", content, flags=re.S)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                parsed = value
                break
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array from LLM response.")
    return [item for item in parsed if isinstance(item, dict)]
