"""Phase B / step (b): plain-text connectivity probe.

Sends a minimal chat request through the same adapter the harness uses and checks
that the endpoint answers with the exact token ``API_OK``. Raw output (request
configuration is sanitized by the adapter's trace log) goes to stdout, which the
caller redirects into preflight/b2_text_connectivity.log.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path(os.environ.get("TEAMBENCH_REF", "D:/omp/MainSearch/explore/llm_etm_benchmark/teambench_ref"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from qwen3_adapter import base_host, create_qwen3_adapter, load_endpoint_config  # noqa: E402

PROMPT = "Reply with exactly this token and nothing else: API_OK"


def main() -> int:
    cfg = load_endpoint_config()
    os.environ.setdefault("TEAMBENCH_MAX_RETRIES", "1")  # preflight: surface failures raw
    adapter = create_qwen3_adapter(temperature=0.2)
    print("request_config:")
    print(json.dumps({
        "model": adapter.model,
        "temperature": adapter.temperature,
        "max_tokens": adapter.max_tokens,
        "seed": adapter.seed,
        "enable_thinking": adapter.enable_thinking,
        "base_host": base_host(cfg["base_url"]),
        "env_file": cfg["env_file"],
        "max_retries": adapter.max_retries,
    }, indent=2))

    start = time.time()
    try:
        text = adapter.generate([{"role": "user", "content": PROMPT}])
    except Exception as exc:  # keep the raw failure, do not retry
        print(f"\nERROR after {time.time() - start:.1f}s: {type(exc).__name__}: {exc}")
        return 1
    elapsed = time.time() - start
    print(f"\nraw_response={text!r}")
    print(f"elapsed_sec={elapsed:.2f}")
    print(f"usage={json.dumps(adapter.get_usage())}")
    matched = text.strip() == "API_OK"
    print(f"exact_match_API_OK={matched}")
    return 0 if matched else 2


if __name__ == "__main__":
    raise SystemExit(main())
