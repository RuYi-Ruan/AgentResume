"""Tiny config loader + OpenAI-compatible chat client (SiliconFlow).

Config precedence:
  1. D:/omp/apikey.txt        (first non-comment non-empty line = the key)
  2. D:/omp/.env / env vars   (SILICONFLOW_API_KEY, QWEN_MODEL, QWEN_BASE_URL)
Defaults: base url = https://api.siliconflow.cn/v1
          model    = Qwen/Qwen3.5-9B
Never prints secrets.
"""
from __future__ import annotations

import json
import os
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_BASE = "https://api.siliconflow.cn/v1"


def _read_keyfile():
    p = ROOT / "apikey.txt"
    if not p.exists():
        return ""
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return ""


def load_env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


load_env()
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "") or _read_keyfile()
MODEL = os.environ.get("QWEN_MODEL", DEFAULT_MODEL)
BASE_URL = os.environ.get("QWEN_BASE_URL", DEFAULT_BASE)


class LLMError(RuntimeError):
    pass


def chat(messages, temperature=0.2, max_tokens=600, json_mode=True):
    """One chat completion. Returns assistant text."""
    if not API_KEY:
        raise LLMError("API key not set: put it in D:/omp/apikey.txt or SILICONFLOW_API_KEY")
    url = BASE_URL.rstrip("/") + "/chat/completions"
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},  # Qwen3.x: avoid reasoning-only responses
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    import time

    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < 2:
                time.sleep(4 * (attempt + 1))
    else:
        if isinstance(last, urllib.error.HTTPError):
            raise LLMError(f"HTTP {last.code}: {last.read().decode('utf-8')[:300]}") from last
        raise LLMError(f"request failed after retries: {last!r}") from last
    try:
        return out["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise LLMError(f"unexpected payload: {str(out)[:200]}") from e


def chat_json(messages, temperature=0.2, max_tokens=600, tries=2):
    """chat() with JSON-object parsing; retries once on parse failure."""
    last = None
    for attempt in range(tries * 2):
        try:
            text = chat(messages, temperature=temperature, max_tokens=max_tokens)
        except LLMError as e:
            if attempt % 2 == 1:
                raise
            # some endpoints reject response_format; retry without JSON mode
            text = chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=False)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            last = e
    raise LLMError(f"could not parse JSON after {tries} tries: {last}; text={text[:200]}")
