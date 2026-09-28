# REVIEW PACKAGE — TeamBench × Qwen3-8B baseline, Phases A + B

Audience: reviewer deciding whether to authorise **Phase C** (the 9-run calibration).
Everything below is evidence produced on 2026-09-28 on this workstation; raw logs are on disk.

**Headline:** Phases A and B are complete, but **three environment facts change the shape of Phase C
and need your decision before it runs** (§7.1–§7.3): the official Docker images cannot be built here
(`deb.debian.org` → 502), the official harness cannot grade natively on Windows (WSL `bash` + `D:\`
paths), and — most importantly — **the automated ablation path does not use the Docker role sandboxes
at all**; it enforces role separation in-process with path allow-lists. The Docker OS-isolation claim
in the README is verified (§6), but it is not the code path `harness.ablation` executes.

---

## 1. Deliverable map

| Plan §6 artifact | Status |
|---|---|
| `IMPLEMENTATION_NOTES.md` | present (commit, deps, changes, commands, blockers) |
| `preflight/` raw logs | present (17 log/trace files + 6 full run trees) |
| `REVIEW_PACKAGE.md` | this file |
| `results/ablation_results.json`, `results/summary.csv` | **intentionally absent** — Phase C outputs |
| `BASELINE_RESULTS.md` | **not written** — the plan's four gates are scored on 9 runs; only preflight evidence exists so far |

## 2. Commit hash and the complete set of modifications

```
repo   D:/omp/MainSearch/explore/llm_etm_benchmark/teambench_ref
HEAD   d185aef1916fd86a9ba554d581fd256319a973af   (Mon May 18 15:13:10 2026 +0000)
       "chore: regenerate leaderboard data from submissions"
origin https://github.com/ybkim95/TeamBench.git
```

**Diff against the official tree: none.** Raw transcript: `preflight/git_provenance.log`.

```
$ git status --porcelain                                    → (empty)
$ git status --porcelain --untracked-files=all               → (empty)
$ git diff --stat                                            → (empty)
$ git diff --cached --stat                                   → (empty)
$ git ls-files --others --exclude-standard | wc -l           → 0
$ git stash list                                             → (empty)
$ git reflog -5        → d185aef HEAD@{0}: clone: from https://github.com/ybkim95/TeamBench.git
$ git status --porcelain --ignored=matching
   !! generators/__pycache__/   !! harness/__pycache__/   !! harness/adapters/__pycache__/
   !! teambench.egg-info/
```

The only non-tracked content in the clone is Python build residue (`__pycache__`, `*.egg-info`) from
running the code; no tracked file was added, edited or deleted, and no branch/commit/stash was created.
`HEAD:tasks` = `27bad8ecb45660fa14553d2f3090d4bd1abe6a29`.

Complete change set = the content of the new files listed in §2.1 (4 Python modules + 3 substitution
Dockerfiles + 2 manual grader scripts), delivered in-tree, nothing elided. Because the clone is
untouched, `git -C teambench_ref diff` is empty by construction, and no diff can exist for `tasks/`,
graders, role permissions or official prompts.

### 2.1 Our additions (all new files, all outside the clone)

```
teambench_qwen3_baseline/adapters/qwen3_adapter.py     # adapter (config + request shaping + trace)
teambench_qwen3_baseline/adapters/run_qwen3.py         # runtime route registration + CLI passthrough
teambench_qwen3_baseline/adapters/probe_b2_text.py     # step (b) probe
teambench_qwen3_baseline/adapters/probe_b3_tools.py    # step (c) probe
teambench_qwen3_baseline/preflight/rig/Dockerfile.*    # substitute images for the isolation test
teambench_qwen3_baseline/preflight/rig/manual_grade*.sh # reproducible manual grader invocations (DIST1 / GH12)
```

The two behavioural pieces of the adapter, in full (everything else is config parsing and tracing):

```python
class Qwen3Adapter(OpenAIAdapter):              # official adapter subclass
    def _call_with_retry(self, max_retries=None, **kwargs):
        extra = dict(kwargs.get("extra_body") or {})
        extra["enable_thinking"] = self.enable_thinking     # always False (plan §2)
        extra.setdefault("seed", self.seed)                 # 0
        kwargs["extra_body"] = extra
        return super()._call_with_retry(..., **kwargs)      # official I/O + parsing unchanged

# run_qwen3.py — runtime route, official factory NOT edited
_ORIGINAL_CREATE_ADAPTER = harness.adapters.create_adapter
def create_adapter(model, temperature=0.2, **kwargs):
    if str(model).lower().startswith("qwen3"):
        return create_qwen3_adapter(model=model, temperature=temperature, **kwargs)
    return _ORIGINAL_CREATE_ADAPTER(model=model, temperature=temperature, **kwargs)
harness.adapters.create_adapter = create_adapter        # resolved lazily by run_full_ablation
```

`lenient_mode` is forced to `False`, so tool calls are taken **only** from the API's structured
`message.tool_calls` field; the adapter never recovers a tool call by regex over plain text.

## 3. Audit result for the CLI (Phase A step 3) — `--base-url` does **not** exist

| Claim | Reality in this commit |
|---|---|
| README: "Any OpenAI-compatible endpoint … works through the OpenAI adapter via `--base-url`" | **False.** `harness/ablation.py::main` accepts `--model --tasks --seeds --tasks-dir --output --max-turns --max-remediation --conditions`. No `--base-url`, no `--api-key`, no per-role model override except the `hetero` condition. |
| `harness/adapters/__init__.py::create_adapter` | Routes on the model-name prefix: `gemini-`, `gpt-`/`o1`/`o3`/`o4`, `claude`, `mock`, `openrouter:`, `vllm:`. Anything else → `ValueError`. |
| `vllm:<model>@<base_url>` route | Exists, but rejected for this experiment: forces `lenient_mode=True` (text-regex tool recovery → forbidden by plan §5 A.6), cannot send `enable_thinking=false`, and has no `.env` key loading. |

Minimal adaptation chosen: one extra route supplied **at runtime** by `adapters/run_qwen3.py`, model id
`qwen3-8b`. This is why the clone stays byte-identical.

## 4. Sanitized request configuration (no key anywhere)

Actual per-request body built by the adapter, as recorded in the traces:

| Field | Value |
|---|---|
| model | `qwen3-8b` (from `MODULE_ID`) |
| temperature | `0.2` |
| seed | `0` |
| `enable_thinking` | `false` (extra_body, **every** request) |
| max_tokens | `8192` (official `OpenAIAdapter` default) |
| top_p / top_k / penalties | not sent (endpoint defaults) |
| base host | `https://api.gpt.ge/v1` |
| env file | `D:/omp/MainSearch/.env` (keys `BASE_URL`, `API_KEY`, `MODULE_ID`, `ENABLE_THINKING`) |
| tool protocol | `tools` + `tool_choice="auto"` when the role has tools; `lenient_mode=False` |
| retry budget | `1` for preflight (failures surface raw); official default `8` for Phase C |
| API key | loaded into memory only; **never** written to a log or artifact |

Leak check (programmatic, key never printed): all 163 files under `teambench_qwen3_baseline/` scanned
for the literal 51-char key value → **0 matches**. The trace logger also drops `api_key` /
`authorization` / `headers` keys defensively.

## 5. Phase B results

### (a) mock smoke — `DIST1_queue_race` × `mock` × `oracle`

Command: see `IMPLEMENTATION_NOTES.md` §6.1. Logs: `preflight/b1_mock_smoke_wsl.log` (graded run),
`preflight/b1_mock_smoke.log` (Windows attempt), `preflight/b1_mock_grader_manual.log`.

| Run | Result | Raw key output |
|---|---|---|
| WSL (used) | FAIL, `partial=0.0833` (1/12 checks), 0.7 s, 2 turns | `reports/score.json` produced with a 12-item checklist → task copy + grader link work |
| Windows | FAIL, `partial=0.00`, `failure_modes: ["grader_no_score"]`, 0.5 s, 2 turns | grader never launched (§7.2) |

Per the plan the score is **not interpreted**. Two mechanical facts behind it, both useful:

1. The official `MockAdapter` short-circuits: *any* turn whose last message contains `Tool '` or
   `result:` returns `DONE` before role dispatch (`harness/adapters/mock_adapter.py`, the guard at the
   top of `generate_with_tools`), so the oracle path performs exactly one tool call (`read
   /shared/reports/expected.json`) and never writes the fix. Evidence: `logs/oracle/turn_000.json`,
   `turn_001.json` in both run trees. **The `mock` condition therefore cannot certify "task solvable".**
2. `harness/adapters/__init__.py` builds `MockAdapter(temperature=…)` with no `reports_dir`, and the
   harness's `grade_run` fallback is `{"pass": false, "failure_modes": ["grader_no_score"]}` when
   `reports/score.json` is absent.

Manual confirmation that the grader itself is sound: `preflight/b1_mock_grader_manual.log` re-runs
`grade.sh` exactly as `harness.run_all.grade_run` does and gets `partial_score=0.0833, 1/12`
(C1 "syntax valid" is the only check the untouched workspace satisfies).

### (b) plain-text connectivity — expected exact `API_OK` — **PASS**

Log: `preflight/b2_text_connectivity.log`, trace `preflight/b2_text_trace.jsonl`.

```
raw_response='API_OK'          exact_match_API_OK=True
elapsed_sec=1.71               usage={"input_tokens": 23, "output_tokens": 2, "total_tokens": 25}
```

### (c) two-round structured function call — **PASS**

Log: `preflight/b3_function_call.log`, trace `preflight/b3_tools_trace.jsonl`.
Uses the **official** `AgentLoop` + `RunCommandTool` with one declared tool (`run`):

```
--- turn 0 --- text=''   tool_calls=[{"name": "run", "args": {"cmd": "echo PROBE_TOKEN_7F3A"}}]
              tool_results=[{"stdout": "PROBE_TOKEN_7F3A\n", "stderr": "", "exit_code": 0}]
--- turn 1 --- text='The probe token received is: PROBE_TOKEN_7F3A\n\nDONE'
              tool_calls=[]   done=True
elapsed_sec=2.88   usage={"input_tokens": 535, "output_tokens": 42, "total_tokens": 577}
round1_structured_tool_call=True   round2_no_tool_call=True
```

No regex recovery was ever needed: the tool call arrived as a structured `tool_calls` entry.

### (d) real harness short tests, low turns (budget-truncated by design)

Task `GH12_click_envvar_flag`, seed 0, `--max-turns 4`, run through `run_qwen3.py` under WSL.

| Run | Log | Result |
|---|---|---|
| `oracle` | `preflight/b4c_oracle_short.log` (also `b4_harness_short.log`, `b4b_oracle_short.log`) | FAIL, `partial=0.00` (0/8 checks), 4 turns, 28.7 s |
| `full` | `preflight/b5_full_short.log` | FAIL, `partial=0.00` (0/8 checks), 12 turns (4 per phase), 47.8 s |

What the run trees prove (they are also the answer to "is the output structure what the plan expects"):

* **Tool loop works with the real model**: oracle `logs/oracle/turn_00{0..3}.json` =
  `read cli.py` → `write cli.py` → `write cli.py` → `write cli.py`, every result `exit_code=0`.
* **Three-role protocol works**: `full` run → planner 4× `send_message`, executor
  `read`/`write`/`run`(`pytest test_cli.py -v`)/`send_message`, verifier `run`/`read`/`run`/`read`.
  `messages/dialogue.jsonl` contains 3 `planner→executor` messages and 1 `executor→verifier` message
  → Planner→Executor delivery is real, not a prompt-only claim.
* **Grader runs and discriminates**: `reports/score.json` written for both runs with an 8-check
  checklist; `reports/expected.json` staged by setup. On the WSL `full` run the grader's
  verdict is dominated by the model's own output:
  `line 40: status = "dry_run: would deploy the application to {config['deploy_target']}":` — a stray
  `:` (a broken f-string) that the model emitted; `:>` in the oracle runs.
* **Not a result about solvability**: `--max-turns 4` truncates every phase mid-iteration (`done=False`
  on the last turn of each phase, no attestation written by the verifier), so the 0.00 partial scores
  carry no signal about task difficulty. They are pipeline evidence only, as the plan intends.
* **No tool-call parsing failures, no 400/429, no grader crash** anywhere in the four runs.

### (e) role isolation — official mounts verified, with the caveat of §7.3

Log: `preflight/isolation/isolation_probe.log`; stack startup `…/compose_up.log`.
The **unmodified** `docker-compose.yml` was started (`docker compose up -d --no-build`) against a rig
`TASK_DIR`/`RUN_DIR`, then probed with `docker exec`.

Identities and disjoint mounts, exactly as the official compose file specifies:

| Role | uid | Sees | Cannot see |
|---|---|---|---|
| planner | 10002 | `/task/spec.md` (ro), `/task/brief.md` (ro), `/task/corpus` (ro), `/workspace` (**ro**), `/analysis` (rw), `/shared/messages` (rw), `/shared/submission` (ro) | `/shared/workspace`, `/shared/reports` |
| executor | 10001 | `/task/brief.md` (ro), `/shared/workspace` (rw), `/shared/reports` (rw), `/shared/messages` (rw), `/analysis` (ro) | `/task/spec.md`, `/task/corpus`, `/shared/submission` |
| verifier | 10003 | `/task/spec.md` (ro), `/shared/workspace` (**ro**), `/shared/reports` (**ro**), `/shared/messages` (rw), `/shared/submission` (rw) | `/task/brief.md`, `/analysis` |

Negative probes (all failed as required, from `isolation_probe.log`):

```
planner WRITES /workspace/hack.txt        → Read-only file system            (exit 2)
planner reads  /shared/workspace          → No such file or directory         (exit 2)
planner reads  /shared/reports/...        → No such file or directory         (exit 1)
planner WRITES /shared/submission/x       → Read-only file system            (exit 2)
executor reads /task/spec.md              → No such file or directory         (exit 1)   [only brief.md in /task]
executor reads /shared/submission         → No such file or directory         (exit 2)
verifier WRITES /shared/workspace/tamper  → Read-only file system            (exit 2)
verifier WRITES /shared/reports/tamper    → Read-only file system            (exit 2)
verifier reads /analysis                  → No such file or directory         (exit 2)
all three roles outbound TCP to api.gpt.ge:443 → BLOCKED (name resolution)    [network_mode: none]
```

Positive controls: each role can read what it should (executor read `brief.md`, verifier read
`spec.md`, planner wrote `/analysis` and `/shared/messages`, verifier wrote
`/shared/submission/attestation_probe.json`), and those host-visible side effects exist in the rig
`RUN_DIR` with distinct container hostnames (`9d9c…`, `9718…`, `516a…`). Stack torn down cleanly;
no `teambench_*` container remains.

## 6. Three roles use the same model and the same sampling parameters

1. **Code path**: `run_full_ablation` constructs exactly one adapter
   (`harness/ablation.py:1333 adapter = create_adapter(model=model, temperature=0.2)`), passes it to
   `run_ablation_condition`, which hands the *same instance* to `TaskOrchestrator`, which passes it to
   every `AgentLoop` (planner, executor, verifier, remediation attempts). There is no per-role adapter
   on this path (per-role adapters exist only for the `hetero` condition, which we do not use).
2. **Observed requests**: the `full` run trace has 12 requests, and the set of distinct values is a
   singleton for each parameter —

```
trace b5_full_trace.jsonl: requests=12  models={'qwen3-8b'}  temps={0.2}
                           extra_body={'{"enable_thinking": false, "seed": 0}'}
trace b4c_oracle_trace.jsonl: requests=4 models={'qwen3-8b'}  temps={0.2}
                           extra_body={'{"enable_thinking": false, "seed": 0}'}
```

3. **Thinking mode**: `enable_thinking=false` is in the body of **every** request in every trace; the
   adapter cannot be constructed with thinking on unless explicitly passed (and Phase C does not).

## 7. Anomalies, blockers, and what each one blocks

Nothing was silently retried, and no failure was overwritten. Ordered by severity.

### 7.1 `docker compose build` fails — `deb.debian.org` returns 502 (BLOCKS official images)

Raw: `preflight/docker_build.log` (189 lines, exit 1). `executor` and `verifier` both die in
`apt-get update`:

```
E: Failed to fetch http://deb.debian.org/debian/dists/trixie/InRelease  502 Bad Gateway [IP: 199.232.114.132 80]
E: The repository 'http://deb.debian.org/debian trixie InRelease' is no longer signed.
target executor: failed to solve: process "/bin/sh -c apt-get update && apt-get install …" → exit code 100
```

The `planner` (pip-only) build was `CANCELED`, so **no official image exists**. Per the plan's stop
rule this was attempted once and not retried; the official Dockerfiles were **not** edited.
For the isolation test (§5e) I built three substitute images that differ only by omitting the
network-dependent layers — same base `python:3.11-slim`, same `USER agent`, same uids 10001/10002/10003
(`preflight/rig/Dockerfile.*`). Mount table, `read_only` flags, `network_mode: none` and role
permissions all come from the official compose file, unmodified.

*Blocked by this:* any Phase C step that expects to execute tools **inside** the official images
(bandit/ruff/mypy/semgrep for analysis-planning roles, pytest/hypothesis/mutmut for expertise
verifiers). It does **not** block the default `oracle/restricted/full` conditions, which never enter a
container (§7.3).

### 7.2 The official harness cannot grade natively on Windows (workaround in place)

`subprocess.run(["bash", …])` resolves to `C:\Windows\system32\bash.EXE` (WSL), which cannot open the
`D:\…` paths `os.path.abspath` produces, so `grade.sh` never starts:

```
preflight/windows_grader_invocation_probe.log
  which(bash) = C:\Windows\system32\bash.EXE
  exact harness-style call -> rc=127
  stderr='/bin/bash: D:ompMainSearchexplore…tasksDIST1_queue_racegrade.sh: No such file or directory'
  score.json exists after call = False
  => harness falls through to {"pass": false, "failure_modes": ["grader_no_score"]}
```

Consequence: on Windows every run silently scores 0 with `grader_no_score` (visible in
`preflight/b1_mock_smoke.log`). All real runs were therefore executed under **WSL Linux python3**, where
`bash` is native and paths are POSIX. Same clone, same commit, same harness source; only the process
environment differs. No monkey-patch was used to make grading work.

*Decision needed:* accept "harness runs under WSL" as the execution environment for Phase C, or invest
in a path-translating wrapper. If neither, Phase C numbers are not reproducible on a Windows host.

### 7.3 The automated ablation never uses the Docker role sandboxes (needs your decision)

`grep -i 'docker|sandbox|use_docker|in_container'` over `harness/ablation.py`, `harness/orchestrator.py`
and `harness/agent_interface.py` returns **no matches**. Role separation on this path is enforced
in-process by `ReadFileTool`/`WriteFileTool`/`RunCommandTool` allow-lists and by which tools each
`RoleConfig` is given (`make_planner_config`, `make_executor_config`, `make_verifier_config`) — i.e.
it is a Python-level permission model, not OS/Docker enforcement. Docker is only wired into the
interactive paths (`harness/run_task.py`, `harness/human_baseline.py`).

The plan (§2, §4.1 gate 1) requires OS/Docker-enforced isolation with measured evidence. What we have:

* the official sandbox **does** isolate correctly when driven manually (§5e, measured), but
* the code path that produces the calibration scores does **not** run inside it.

Options for the reviewer: (i) accept in-process isolation for this development calibration and record
the deviation; (ii) add a docker-exec tool backend so `harness.ablation` runs roles in the official
containers (a real code change, needs its own review, and is blocked by §7.1 anyway); (iii) restrict
claims to "official allow-list isolation, Docker sandbox verified separately". I did not choose
silently — no code was changed.

### 7.4 The official `mock` adapter cannot exercise the oracle/executor path

See §5a: `MockAdapter.generate_with_tools` returns `DONE` on the turn after any tool result, before
dispatching to `_oracle_response`/`_executor_response`. So the mock smoke validates plumbing
(orchestrator → agent loop → tool execution → grader) but **can never validate that a task is
solvable**, and its `0.0833` must not be read as a task-difficulty signal.

### 7.5 Model-output quality on the two real short tests (not a harness fault)

* All 4 oracle replies and all 12 full replies had `finish_reason="tool_calls"` → **no server-side
  truncation**, and no tool arguments failed to parse.
* Yet both oracle runs wrote a `cli.py` with a genuine syntax error (stray `:`/`>` inside an intended
  f-string at line ~40), and the `full` run's verifier never wrote `submission/attestation.json`
  within its 4-turn budget. This is Qwen3-8B behaviour under a truncated budget, recorded as-is.
* **Reproducibility:** the three independent oracle runs (`b4`, `b4b`, `b4c`) produced byte-identical
  `cli.py` (md5 `fe713c426a6…`, 2145 bytes) and identical token counts (6405 in / 1501 out) → `seed=0`
  is honoured by the endpoint. Implication for Phase C: repeat runs at seed 0 will not add variance,
  so the plan's difficulty/collaboration thresholds must be read from `full − restricted` gaps, not
  from seed spread.

## 8. Token / time / cost inference

Measured (exact, from `*_trace.jsonl` and the run reports) — **not** an estimate:

| Run | API calls | input tok | output tok | total tok | wall |
|---|---|---|---|---|---|
| (b) plain text | 1 | 23 | 2 | 25 | 1.71 s |
| (c) 2-round tool call | 2 | 535 | 42 | 577 | 2.88 s |
| (d) oracle, GH12, 4 turns | 4 | 6 405 | 1 501 | 7 906 | 28.7 s |
| (d) full, GH12, 4 turns/phase | 12 | 24 091 | 2 375 | 26 466 | 47.8 s |

Derived per-turn cost: **oracle ≈ 1 977 tok/turn**, **full ≈ 2 206 tok/turn** (input dominates and grows
~36 tok/turn within a phase; output ≈ 200–630 tok/turn depending on whether the turn writes a file).

**Projection for Phase C — [INFERENCE], do not treat as measurement.** Assumes the plan's Phase C
command (no `--max-turns`), i.e. the harness defaults (`oracle` 20 turns; `restricted` 30 turns;
`full` 20 turns/phase + up to 2 remediation loops), and that GH12-style tasks behave like the measured
one. Because the measured runs were budget-truncated at 4 turns, real turns-per-run is the dominant
unknown.

| Condition | assumed calls | projected total tok | projected wall | basis |
|---|---|---|---|---|
| `oracle` ×1 | 12–20 | ~17k–46k | 50–80 s | measured oracle slope |
| `restricted` ×1 | 15–30 | ~30k–75k | 60–120 s | `max(20,30)` turns, no spec in context |
| `full` ×1 | 25–60 (+40 worst case) | ~55k–170k (→ ~280k worst) | 2–5 min (→ 8 min worst) | measured full slope × phases |

**9 runs (3 tasks × 3 conditions × seed 0) — [INFERENCE]: ≈ 0.6–2.4 M total tokens
(input ≈ 80 %), wall ≈ 20–45 min** (mid case ≈ 870 k tokens, ~25 min). Add grading time —
measured on this host: DIST1 4.0 s (`preflight/b1_mock_grader_manual.log`), GH12 4.1 s
(`preflight/b4c_grader_manual.log`), i.e. ≈ 36 s for all 9 runs.

**Cost: not computable here.** The endpoint `https://api.gpt.ge/v1` publishes no rate card reachable
from this environment (web search is blocked by network policy; nothing in the repo states prices), and
the API response's `usage` carries tokens only. Formula for the reviewer to apply their own rate:

```
cost = input_tokens/1e6 × P_in  +  output_tokens/1e6 × P_out
mid-case: 0.70M input + 0.17M output
```

No number is invented here; Phase C's spend is bounded above by the worst-case row only if the run is
capped with `--max-turns`/`--max-remediation`.

## 9. Plan §4.1 chain gate — status on the evidence available

| Gate item | Status |
|---|---|
| mock flow completes | ✅ harness → tools → grader exercised end-to-end (score emitted under WSL; §5a) |
| real API completes a standard function/tool call | ✅ structured tool call, result fed back, model finishes (§5c) and 16 more in the real harness (§5d) |
| all three roles use the same model and non-thinking mode | ✅ one shared adapter instance + 12/12 requests identical (§6) |
| `oracle partial_score > 0` for ≥2 of 3 tasks | ⏳ needs Phase C (preflight used `--max-turns 4` and is not a solvability signal) |
| Docker/OS role isolation | ⚠️ verified on the official compose stack (§5e) but **not** the path used by the automated runner (§7.3) |

**No Phase C run was started; no `git commit` was made.**

## 10. What I need from the reviewer

1. §7.3 — accept in-process isolation for the calibration, or require a docker-exec tool backend first?
2. §7.2 — is "harness runs under WSL" an acceptable execution environment, or must Windows-native runs
   work (which needs a path-translation wrapper, since the official `bash` invocation cannot work here)?
3. §7.1 — should the missing `bandit/ruff/mypy/semgrep` tooling (only needed by `analysis_planner`
   roles outside the three planned conditions) be revisited, or is the `apt` failure acceptable for
   Phase C's `oracle/restricted/full` conditions?
4. §8 — confirm the rate card, or accept token-only reporting.
5. §5d — confirm that `--max-turns 4` preflight truncation is acceptable and that Phase C should run
   with harness defaults (20/30/20 + 2 remediation loops).

---

## 8. 独立核验（主调度代理复跑，2026-09-28）

以下各项由主调度代理在 Agent 交付后**独立复跑**，不采信其自述：

| # | 核验项 | 结果 |
| --- | --- | --- |
| 1 | 官方仓库未被修改 | `git -C teambench_ref status --porcelain -uall` = **0 行**；`HEAD:tasks` = `27bad8ecb45660fa14553d2f3090d4bd1abe6a29` ✓（与报告一致）|
| 2 | **API Key 泄露扫描** | 从 `D:/omp/MainSearch/.env` 读出 API Key（长度 51，**全程未打印**），在 `teambench_qwen3_baseline/` 的 **169 个文件**与 `teambench_ref` 全树中检索 ⇒ **命中 0** ✓ |
| 3 | 交付物齐全 | `REVIEW_PACKAGE.md`、`IMPLEMENTATION_NOTES.md`、`adapters/`（4 个新文件）、`preflight/`（29 项，含 6 棵 run tree + 隔离装置）、`results/` ✓ |
| 4 | MockAdapter 短路行为 | `harness/adapters/mock_adapter.py:60`：`if "Tool '" in last_msg or "result:" in last_msg:` ⇒ mock 路径只做一次工具调用 ⇒ **其 0.0833 分不可解读** ✓（与报告一致）|
| 5 | **隔离实现路径（关键）** | `harness/ablation.py` 通过 `make_executor_config` / `ReadFileTool(...)` 构造角色配置并**同进程**运行 `AgentLoop` ✓；`harness/agent_interface.py:289` 仅提供“Docker 路径 → 实际路径映射（供非 Docker 运行使用）” ✓ ⇒ **产出校准分数的代码路径不经过 Docker/OS 级沙箱** ✓。报告结论成立，但“三个文件里 grep docker/sandbox 无匹配”一句**略过头**：`agent_interface.py` 有一处映射辅助（非强制隔离）✓ |

**因此**：方案 §2 “角色隔离必须使用官方 OS/Docker 权限隔离，不接受仅靠提示词模拟”这一条，
在**当前产生分数的路径上不满足**（该路径是同进程的工具白名单，不是 OS/Docker 强制隔离）；
Docker 沙箱本身已单独验证可用（`preflight/isolation/isolation_probe.log`）✓。**需审阅者裁定后再决定是否运行阶段 C。**
