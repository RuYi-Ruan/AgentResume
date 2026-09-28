"""Minimal third-party OpenAI-compatible adapter for the TeamBench harness.

Purpose
-------
TeamBench ships adapters for Gemini / OpenAI / Anthropic / mock and a `vllm:<model>@<base_url>`
prefix. None of them can talk to the third-party Qwen3-8B endpoint used for this
calibration, because that endpoint needs:

  * a custom base URL and API key that live in ``D:/omp/MainSearch/.env``
  * an explicit ``enable_thinking=false`` on every single request
  * a fixed ``seed``
  * the strict structured tool-call protocol (no regex recovery from plain text)

This module supplies exactly that, as a *new file*; the official TeamBench tree is
left byte-identical (see REVIEW_PACKAGE.md, `git -C teambench_ref status`).

Design constraints (from TEAMBench_QWEN3_8B_BASELINE_PLAN.md):
  * Subclasses the official ``harness.adapters.openai_adapter.OpenAIAdapter`` so
    request building, response parsing and tool-call extraction stay official.
  * ``lenient_mode=False`` -> tool calls are only ever taken from the API's
    structured ``message.tool_calls`` field, never guessed from text.
  * The API key is never printed, logged or written to disk: only the base *host*
    is reported (``redact_secret`` / ``base_host``).
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from urllib.parse import urlparse

from harness.adapters.openai_adapter import OpenAIAdapter

# Default configuration sources, in order. The first existing file wins.
DEFAULT_ENV_CANDIDATES = (
    "D:/omp/MainSearch/.env",
    "/mnt/d/omp/MainSearch/.env",
)

DEFAULT_MODEL_KEY = "qwen3-8b"


def _parse_env_file(path: pathlib.Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file (quotes optional, no expansion)."""
    cfg: dict[str, str] = {}
    if not path.is_file():
        return cfg
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        cfg[key.strip()] = value
    return cfg


def env_file_path() -> pathlib.Path:
    override = os.environ.get("TEAMBENCH_ENV_FILE")
    candidates = (override,) if override else DEFAULT_ENV_CANDIDATES
    for cand in candidates:
        p = pathlib.Path(cand)
        if p.is_file():
            return p
    return pathlib.Path(candidates[0])


def load_endpoint_config() -> dict:
    """Read BASE_URL / API_KEY / MODULE_ID / ENABLE_THINKING (env overrides file)."""
    path = env_file_path()
    cfg = _parse_env_file(path)
    base_url = os.environ.get("TEAMBENCH_BASE_URL") or cfg.get("BASE_URL", "")
    api_key = os.environ.get("TEAMBENCH_API_KEY") or cfg.get("API_KEY", "")
    model = os.environ.get("TEAMBENCH_MODEL") or cfg.get("MODULE_ID", DEFAULT_MODEL_KEY)
    raw_think = (
        os.environ.get("TEAMBENCH_ENABLE_THINKING")
        or cfg.get("ENABLE_THINKING", "false")
    ).strip().lower()
    return {
        "env_file": str(path),
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        # The plan fixes thinking mode OFF for every request. Anything other than an
        # explicit "true" is treated as false, and the value is asserted below.
        "enable_thinking": raw_think == "true",
    }


def base_host(base_url: str) -> str:
    """Return only scheme+host+path of the endpoint, safe to log."""
    parsed = urlparse(base_url)
    if not parsed.scheme:
        return base_url
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def _max_retries() -> int:
    """Official retry budget (8) unless the operator pins it for preflight runs."""
    try:
        return max(1, int(os.environ.get("TEAMBENCH_MAX_RETRIES", "8")))
    except ValueError:
        return 8


# Role labels as they appear in the official role system prompts
# (harness/agent_interface.py, harness/ablation.py). Used only to *label* the
# trace so audit_privileges.py can attribute adapter-side tool-call records to a
# role; nothing about the request or the response is affected.
_ROLE_PROMPT_MARKERS = (
    ("You are the Planner. You are a static analysis expert", "planner"),
    ("You are the Planner", "planner"),
    ("You are the Executor", "executor"),
    ("You are the Verifier. You verify correctness", "verifier"),
    ("You are the Verifier", "verifier"),
    ("You are a Restricted agent", "restricted"),
    ("You are an Oracle agent", "oracle"),
)


def role_hint(system_prompt: str | None) -> str:
    """Best-effort role label for the system prompt the AgentLoop handed over."""
    prompt = system_prompt or ""
    for marker, role in _ROLE_PROMPT_MARKERS:
        if marker in prompt:
            return role
    return "unknown"


class Qwen3Adapter(OpenAIAdapter):
    """OpenAI-compatible adapter for the fixed qwen3-8b calibration endpoint."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        seed: int = 0,
        enable_thinking: bool | None = None,
        trace_log: str | None = None,
        **kwargs,
    ):
        cfg = load_endpoint_config()
        if not cfg["base_url"]:
            raise ValueError(f"No BASE_URL found (env file: {cfg['env_file']})")
        if not cfg["api_key"]:
            raise ValueError(f"No API_KEY found (env file: {cfg['env_file']})")

        self.cfg = cfg
        self.seed = int(seed)
        # Plan §2: every request carries enable_thinking=false, no exceptions.
        self.enable_thinking = False if enable_thinking is None else bool(enable_thinking)

        # lenient_mode stays False: never recover tool calls by regex over plain text.
        kwargs.pop("lenient_mode", None)
        super().__init__(
            api_key=cfg["api_key"],
            model=model or cfg["model"],
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=cfg["base_url"],
            lenient_mode=False,
            **kwargs,
        )
        self.max_retries = _max_retries()
        self.trace_log = trace_log or os.environ.get("TEAMBENCH_TRACE_LOG") or None
        self._trace_fh = None
        if self.trace_log:
            pathlib.Path(self.trace_log).parent.mkdir(parents=True, exist_ok=True)
            self._trace_fh = open(self.trace_log, "a", encoding="utf-8")
            self._trace({
                "event": "adapter_init",
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
                "enable_thinking": self.enable_thinking,
                "base_host": base_host(cfg["base_url"]),
                "env_file": cfg["env_file"],
                "max_retries": self.max_retries,
                "lenient_mode": self.lenient_mode,
            })

    # ------------------------------------------------------------------
    # Request-shape adaptation
    # ------------------------------------------------------------------

    def _call_with_retry(self, max_retries: int | None = None, **kwargs):
        """Inject enable_thinking/seed into the JSON body, then defer to official I/O."""
        extra = dict(kwargs.get("extra_body") or {})
        extra["enable_thinking"] = self.enable_thinking
        extra.setdefault("seed", self.seed)
        kwargs["extra_body"] = extra
        self._trace({
            "event": "request",
            "model": kwargs.get("model"),
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
            "extra_body": extra,
            "tool_choice": kwargs.get("tool_choice"),
            "n_tools": len(kwargs.get("tools") or []),
            "n_messages": len(kwargs.get("messages") or []),
        })
        response = super()._call_with_retry(
            max_retries=self.max_retries if max_retries is None else max_retries,
            **kwargs,
        )
        try:
            choice = response.choices[0] if response.choices else None
            usage = getattr(response, "usage", None)
            self._trace({
                "event": "api_response_meta",
                # finish_reason == "length" means the server truncated the reply.
                "finish_reason": getattr(choice, "finish_reason", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            })
        except Exception:  # never let diagnostics break a run
            pass
        return response

    def generate_with_tools(self, messages, system_prompt, tools):
        before = self.get_usage()
        resp = super().generate_with_tools(messages, system_prompt, tools)
        after = self.get_usage()
        self._trace({
            "event": "response",
            "role_hint": role_hint(system_prompt),
            "system_prompt_sha1": hashlib.sha1(
                (system_prompt or "").encode("utf-8")).hexdigest()[:12],
            "text": resp.text[:2000],
            "tool_calls": resp.tool_calls,
            "done": resp.done,
            "call_input_tokens": after["input_tokens"] - before["input_tokens"],
            "call_output_tokens": after["output_tokens"] - before["output_tokens"],
            "cum_input_tokens": after["input_tokens"],
            "cum_output_tokens": after["output_tokens"],
        })
        return resp

    def generate(self, messages, **kwargs) -> str:
        text = super().generate(messages, **kwargs)
        self._trace({"event": "response_text", "text": text[:2000]})
        return text

    # ------------------------------------------------------------------
    # Sanitized tracing (never logs the API key or auth headers)
    # ------------------------------------------------------------------

    def _trace(self, payload: dict) -> None:
        if not self._trace_fh:
            return
        safe = {k: v for k, v in payload.items()
                if k.lower() not in {"api_key", "authorization", "headers"}}
        self._trace_fh.write(json.dumps(safe, ensure_ascii=False, default=str) + "\n")
        self._trace_fh.flush()

    def close(self) -> None:
        if self._trace_fh:
            self._trace_fh.close()
            self._trace_fh = None


def create_qwen3_adapter(model: str | None = None, temperature: float = 0.2, **kwargs) -> Qwen3Adapter:
    """Factory matching the official `create_adapter(model, temperature, **kwargs)` shape."""
    return Qwen3Adapter(model=model, temperature=temperature, **kwargs)
