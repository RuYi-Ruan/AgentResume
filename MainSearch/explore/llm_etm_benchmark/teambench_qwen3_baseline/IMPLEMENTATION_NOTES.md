# IMPLEMENTATION_NOTES — TeamBench × Qwen3-8B preflight (Phases A + B)

Scope: baseline plan `../TEAMBench_QWEN3_8B_BASELINE_PLAN.md`, **Phase A (environment + adapter)**
and **Phase B (short tests + review package)** only.
**Phase C (the 9-run calibration) was NOT started. No `git commit` was made anywhere.**

Date: 2026-09-28 · Host: Windows 11 Pro (10.0.22000), x64 · Docker 29.2.1 / Compose v5.0.2 (overlayfs)

---

## 1. Official repository pin

| Item | Value |
|---|---|
| Path | `D:/omp/MainSearch/explore/llm_etm_benchmark/teambench_ref` |
| Commit | `d185aef1916fd86a9ba554d581fd256319a973af` (Mon May 18 15:13:10 2026 +0000, "chore: regenerate leaderboard data from submissions") |
| Remote | `https://github.com/ybkim95/TeamBench.git` |
| Working tree | **clean** — `git status --porcelain` and `git diff` are both empty (see REVIEW_PACKAGE.md §4) |
| `tasks/` tree hash at HEAD | `27bad8ecb45660fa14553d2f3090d4bd1abe6a29` |

All of our code lives in the sibling directory `teambench_qwen3_baseline/`, never inside the clone.

## 2. Python environments and dependencies

Two interpreters are used, both against the same source tree:

| Environment | Role | Version |
|---|---|---|
| `llm_etm_benchmark/.venv` (Windows) | probes, adapter development, isolation rig | Python 3.11.3 |
| WSL `python3` (Ubuntu, user `ilin`) | **runs the official harness end-to-end** (see §5 for why) | Python 3.12.3 |

Dependency install command (Windows, from the repo root):

```powershell
D:/omp/MainSearch/explore/llm_etm_benchmark/.venv/Scripts/python.exe -m pip install -e ".[openai]" pytest
```

WSL (needed because the distro ships no `pip`/`ensurepip`):

```bash
curl -sS -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
python3 /tmp/get-pip.py --user --break-system-packages --quiet
python3 -m pip install --user --break-system-packages --quiet click pyyaml openai pytest
```

Resolved versions (full freezes: `preflight/pip_freeze_win.txt`, `preflight/pip_freeze_wsl.txt`):

| Package | Version | Notes |
|---|---|---|
| teambench | 0.1.0 | editable, `-e git+https://github.com/ybkim95/TeamBench.git@d185aef1916fd86a9ba554d581fd256319a973af` |
| openai | 3.19.2 | official `harness/adapters/openai_adapter.py` imports this |
| click / PyYAML | 8.5.0 / 6.0.3 | `[project] dependencies` |
| pytest | 9.1.1 | for graders that call pytest |
| pydantic / httpx2 / httpcore2 / anyio / jiter / truststore | 2.13.5 / 2.13.1 / 2.13.1 / 4.15.1 / 0.17.0 / 0.10.4 | transitive deps of openai 3.19.2 |

No dependency was pinned or patched by hand; nothing in the official repo was touched to make
an install succeed (no `requires.txt`/`pyproject.toml` edits, no `uv.lock` edits).

## 3. What we added (all new files, all outside the official tree)

| File | Purpose |
|---|---|
| `adapters/qwen3_adapter.py` | Third-party OpenAI-compatible adapter. Reads `BASE_URL`/`API_KEY`/`MODULE_ID`/`ENABLE_THINKING` from `D:/omp/MainSearch/.env`, injects `enable_thinking=false` + `seed=0` into every request body, keeps the official structured tool-call parsing (`lenient_mode=False`), and writes a sanitized JSONL trace (no key, no headers). |
| `adapters/run_qwen3.py` | Runtime registration + CLI passthrough. Adds a `qwen3*` route to `harness.adapters.create_adapter` **in memory** (the official `example_custom_adapter.py` suggests editing that factory; we deliberately do not, so the clone stays byte-identical), then calls `harness.ablation.main()`. |
| `adapters/probe_b2_text.py` | Phase B step (b): plain-text connectivity probe. |
| `adapters/probe_b3_tools.py` | Phase B step (c): two-round tool-call probe driven by the official `AgentLoop`. |
| `preflight/rig/Dockerfile.{planner,executor,verifier}` | Substitute role images for the isolation test only (identical base image + identical non-root uids 10002/10001/10003; network-dependent `apt-get`/`pip` layers omitted). Used because the official build is blocked (§5). |
| `preflight/rig/manual_grade.sh`, `manual_grade_gh12.sh` | Reproduce the harness's exact grader invocation by hand (DIST1 / GH12) for the grader-link and grading-time evidence. |
| `PROTOCOL_REVISION.md` | Phase-C prerequisite: record of the five reviewer rulings + the wording limits on any conclusion, plus the audit's rules and limits. |
| `audit_privileges.py` | Independent privilege audit (per-turn logs + adapter trace); exit code 1 on any medium/high/critical finding, so it can wrap Phase C runs. |
| `preflight/audit_selftest/` | Reverse/forward control fixtures (synthetic run trees) and their raw audit output. |

`adapters/qwen3_adapter.py` additionally labels each trace `response` record with `role_hint` and
`system_prompt_sha1` (trace-only; the request body, tool protocol and parsing are unchanged — see
`PROTOCOL_REVISION.md` §3.4). Nothing in `tasks/`, any `grade.sh`, `harness/agent_interface.py` role
permissions or any official role prompt was modified — verified in REVIEW_PACKAGE.md §4.

## 4. The official harness only needs one thing from us: an adapter route

The audit of `harness/` (Phase A step 3) found:

* `harness/ablation.py` (`main`, `run_full_ablation`) exposes `--model --tasks --seeds --tasks-dir
  --output --max-turns --max-remediation --conditions`. **There is no `--base-url` flag** — the
  README statement "Any OpenAI-compatible endpoint ... works through the OpenAI adapter via
  `--base-url`" is not backed by the CLI in this commit.
* `harness/adapters/__init__.py::create_adapter` routes by model-name prefix only: `gemini-`,
  `gpt-`/`o1`/`o3`/`o4`, `claude`, `mock`, `openrouter:`, `vllm:` — anything else raises
  `ValueError`. The closest existing route, `vllm:<model>@<base_url>`, was rejected because it (a)
  needs the key in the environment, (b) has no way to send `enable_thinking=false`, and (c) forces
  `lenient_mode=True`, which recovers tool calls by regex over plain text — explicitly forbidden by
  the plan (§5 A.6).
* Therefore the minimal adaptation is one extra route, supplied at runtime by
  `adapters/run_qwen3.py`. The `--model` id used everywhere is **`qwen3-8b`** (the value of
  `MODULE_ID`).

Interface kept intact: `create_adapter(model, temperature=0.2, **kwargs)` → `ToolCallAdapter` with
`generate_with_tools(messages, system_prompt, tools) -> AdapterResponse(text, tool_calls, done)`.

## 5. Two environment blockers found (raw evidence kept, nothing silently worked around)

**(1) `docker compose build` fails — `deb.debian.org` is unreachable (HTTP 502).**
Full log: `preflight/docker_build.log` (189 lines, exit 1). Both the `executor` and `verifier`
Dockerfiles start with `apt-get update`, which gets:

```
E: Failed to fetch http://deb.debian.org/debian/dists/trixie/InRelease  502  Bad Gateway [IP: 199.232.114.132 80]
E: The repository 'http://deb.debian.org/debian trixie InRelease' is no longer signed.
```

The `planner` image (pip-only) was `CANCELED` by compose's bake because a sibling service failed, so
**no official image exists**. Per the plan's stop rule this was not retried; the raw log is preserved
as-is. The official Dockerfiles were **not** edited to work around it.

**(2) The official harness cannot grade on Windows as-is.** `subprocess.run(["bash", ...])` resolves
`bash` to `C:\Windows\system32\bash.EXE` (WSL), which cannot read the `D:\...` paths the harness
passes (`os.path.abspath`), so `grade.sh` never runs and every run silently degrades to
`{"pass": false, "failure_modes": ["grader_no_score"]}`. Raw evidence:
`preflight/windows_grader_invocation_probe.log` (`rc=127`, stderr
`/bin/bash: D:ompMainSearchexplore...grade.sh: No such file or directory`) and the first mock smoke
`preflight/b1_mock_smoke.log` (`grader_no_score`).

**Consequence / decision:** the real harness runs are executed under **WSL Linux python3**, where
`bash` is native, paths are POSIX and the graders run exactly as shipped. The official source tree is
identical in both cases (same clone, same commit), so this changes only the process environment, not
the harness. No code was monkey-patched to make grading work.

## 6. Commands

### 6.1 Mock smoke (Phase B step a) — the graded one

```bash
# from the official repo root, inside WSL
cd /mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_ref
PATH="$HOME/.local/bin:$PATH" PYTHONPATH=. python3 -m harness.ablation \
  --model mock --tasks DIST1_queue_race --seeds 0 --conditions oracle \
  --output /mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_qwen3_baseline/preflight/b1_mock_wsl/ablation_results.json
```

### 6.2 Plain-text connectivity (step b)

```powershell
cd D:/omp/MainSearch/explore/llm_etm_benchmark
$env:TEAMBENCH_TRACE_LOG = "D:/omp/MainSearch/explore/llm_etm_benchmark/teambench_qwen3_baseline/preflight/b2_text_trace.jsonl"
./.venv/Scripts/python.exe teambench_qwen3_baseline/adapters/probe_b2_text.py
```

### 6.3 Two-round function call (step c)

```powershell
cd D:/omp/MainSearch/explore/llm_etm_benchmark
$env:TEAMBENCH_TRACE_LOG = "D:/omp/MainSearch/explore/llm_etm_benchmark/teambench_qwen3_baseline/preflight/b3_tools_trace.jsonl"
./.venv/Scripts/python.exe teambench_qwen3_baseline/adapters/probe_b3_tools.py
```

### 6.4 Real harness short tests (step d) — oracle and full, low turns

```bash
# from the official repo root, inside WSL
cd /mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_ref
export PATH="$HOME/.local/bin:$PATH"
export TEAMBENCH_REF=/mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_ref
export TEAMBENCH_MAX_RETRIES=1          # preflight only: surface failures raw (official default is 8)
OUT=/mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_qwen3_baseline/preflight

TEAMBENCH_TRACE_LOG=$OUT/b4c_oracle_trace.jsonl python3 \
  $OUT/../adapters/run_qwen3.py --model qwen3-8b \
  --tasks GH12_click_envvar_flag --seeds 0 --conditions oracle --max-turns 4 \
  --output $OUT/b4c_oracle_short/ablation_results.json

TEAMBENCH_TRACE_LOG=$OUT/b5_full_trace.jsonl python3 \
  $OUT/../adapters/run_qwen3.py --model qwen3-8b \
  --tasks GH12_click_envvar_flag --seeds 0 --conditions full --max-turns 4 --max-remediation 0 \
  --output $OUT/b5_full_short/ablation_results.json
```

### 6.5 Isolation check (step e)

```powershell
# substitute images (official build blocked, see §5) — same base image, same uids
cd D:/omp/MainSearch/explore/llm_etm_benchmark/teambench_ref
$RIG = D:/omp/MainSearch/explore/llm_etm_benchmark/teambench_qwen3_baseline/preflight/rig
docker build -t teambench_ref-planner  -f $RIG/Dockerfile.planner  images/planner
docker build -t teambench_ref-executor -f $RIG/Dockerfile.executor images/executor
docker build -t teambench_ref-verifier -f $RIG/Dockerfile.verifier images/verifier

# the OFFICIAL compose file, unmodified, with a rig task/run dir
$ISO = D:/omp/MainSearch/explore/llm_etm_benchmark/teambench_qwen3_baseline/preflight/isolation
$env:TASK_DIR="$ISO/task"; $env:RUN_DIR="$ISO/run"; $env:CORPUS_DIR="$ISO/task/corpus"
docker compose up -d --no-build          # then: docker exec ... probes (see isolation_probe.log)
docker compose down
```

### 6.6 Phase C (for review only — **not executed**)

```bash
cd /mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_ref
python3 ../teambench_qwen3_baseline/adapters/run_qwen3.py \
  --model qwen3-8b \
  --tasks GH12_click_envvar_flag SPEC5_config_system CROSS1_api_contract \
  --seeds 0 \
  --conditions oracle restricted full \
  --output ../teambench_qwen3_baseline/results/ablation_results.json
```

`teambench_qwen3_baseline/results/` is intentionally empty: it is reserved for the Phase C artifacts
(`ablation_results.json`, `summary.csv`).

## 7. Untouched by design

* No `git commit`, no branch, no stash — in any repository.
* No modification to `tasks/`, `graders`, role permission configs (`images/*/Dockerfile`,
  `docker-compose.yml`, `harness/agent_interface.py` role configs) or official role prompts
  (`harness/orchestrator.py` prompts) — proof in REVIEW_PACKAGE.md §4.
* No ETM/memory/reflection/world-model code, per plan §7.
* Phase C (`9` runs) not started.
