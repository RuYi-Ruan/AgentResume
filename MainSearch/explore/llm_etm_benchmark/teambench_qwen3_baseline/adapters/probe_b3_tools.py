"""Phase B / step (c): two-round function-call probe through the OFFICIAL agent loop.

Round 1: the model must emit a structured tool call for the official `run` tool.
Round 2: the locally executed tool result is fed back; the model must finish
         without requesting another tool.

Uses harness.agent_loop.AgentLoop + harness.agent_interface.RunCommandTool
unmodified, so this exercises the same tool declarations, the same response
parsing and the same feedback format as a real TeamBench run.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path(os.environ.get("TEAMBENCH_REF", "D:/omp/MainSearch/explore/llm_etm_benchmark/teambench_ref"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from qwen3_adapter import base_host, create_qwen3_adapter, load_endpoint_config  # noqa: E402
from harness.agent_interface import RoleConfig, RunCommandTool  # noqa: E402
from harness.agent_loop import AgentLoop  # noqa: E402

SYSTEM_PROMPT = (
    "You are a probe agent. You have exactly one tool: `run`, which executes a shell "
    "command and returns its stdout. You MUST use the tool — do not answer from memory. "
    "After you receive the tool result, reply with DONE."
)
PROMPT = (
    "Call the `run` tool with cmd='echo PROBE_TOKEN_7F3A' to print the probe token, "
    "then report the token you received and finish."
)


def main() -> int:
    cfg = load_endpoint_config()
    os.environ.setdefault("TEAMBENCH_MAX_RETRIES", "1")
    adapter = create_qwen3_adapter(temperature=0.2)

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="teambench_probe_"))
    messages_dir = workdir / "messages"
    messages_dir.mkdir(parents=True, exist_ok=True)

    config = RoleConfig(
        role="probe",
        system_prompt=SYSTEM_PROMPT,
        tools=[RunCommandTool(cwd=str(workdir), allowed=True)],
    )
    loop = AgentLoop(
        role_config=config,
        adapter=adapter,
        messages_dir=str(messages_dir),
        log_dir=str(workdir / "logs"),
        max_turns=4,
    )

    print("request_config:")
    print(json.dumps({
        "model": adapter.model,
        "temperature": adapter.temperature,
        "max_tokens": adapter.max_tokens,
        "seed": adapter.seed,
        "enable_thinking": adapter.enable_thinking,
        "base_host": base_host(cfg["base_url"]),
        "max_retries": adapter.max_retries,
        "tool_declared": "run",
    }, indent=2))

    start = time.time()
    try:
        turns = loop.run(PROMPT)
    except Exception as exc:
        print(f"\nERROR after {time.time() - start:.1f}s: {type(exc).__name__}: {exc}")
        return 1
    elapsed = time.time() - start

    for turn in turns:
        print(f"\n--- turn {turn.turn} ---")
        print(f"text={turn.text!r}")
        print(f"tool_calls={json.dumps(turn.tool_calls)}")
        print(f"tool_results={json.dumps(turn.tool_results)}")
        print(f"done={turn.done}")

    print(f"\nelapsed_sec={elapsed:.2f}")
    print(f"usage={json.dumps(adapter.get_usage())}")
    (workdir / "turns.json").write_text(
        json.dumps([dataclasses.asdict(t) for t in turns], indent=2, default=str),
        encoding="utf-8",
    )
    print(f"turns_json={workdir / 'turns.json'}")

    round1_tool = any(
        tc.get("name") == "run" for tc in (turns[0].tool_calls if turns else [])
    )
    round2_clean = bool(len(turns) >= 2 and not turns[1].tool_calls)
    print(f"round1_structured_tool_call={round1_tool}")
    print(f"round2_no_tool_call={round2_clean}")
    return 0 if (round1_tool and round2_clean) else 2


if __name__ == "__main__":
    raise SystemExit(main())
