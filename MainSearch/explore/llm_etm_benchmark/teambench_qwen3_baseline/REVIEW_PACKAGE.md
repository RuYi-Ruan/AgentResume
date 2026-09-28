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

---

## 11. 阶段 C 前置补齐（协议修订 + 越权审计，2026-09-28）

本节是本轮新增交付。**未运行阶段 C，未做 `git commit`，未改官方仓库任何文件。**

新增/修改的文件：

| 文件 | 内容 |
| --- | --- |
| `PROTOCOL_REVISION.md` | **五条裁定的正式记录** + **写死的结论表述限制**（允许/禁止清单）+ 审计器说明与阶段 C 开跑检查表 |
| `audit_privileges.py` | **独立越权审计器**（主来源：逐 turn 日志；次来源：适配器 trace），有违规 ⇒ **退出码 1** |
| `adapters/qwen3_adapter.py` | **仅 trace 层**新增 `role_hint` + `system_prompt_sha1` 两个字段（详见 §11.4） |
| `preflight/audit_selftest/` | 反向/正向对照夹具与实测日志（合成 run 树，非真实运行） |

### 11.1 五条裁定（摘要，全文见 `PROTOCOL_REVISION.md` §1）

| # | 裁定 | 对本报告的影响 |
| --- | --- | --- |
| 1 | **接受官方进程内路径，仅限本次开发校准**；结论只能写「**官方 harness 下的三角色协作收益**」；**不得**声称 OS 强隔离；正式 ETM 实验前另解决 Docker 隔离 | §7.3 的处理方式就是「记录偏离、不宣称强隔离」 |
| 2 | **WSL 为阶段 C 固定运行环境**；Windows 原生结果**无效、不再兼容** | §7.2 的路径翻译包装器**不再做** |
| 3 | `apt` 工具链缺失**不阻塞**本轮 `oracle/restricted/full`，**不再处理** | §7.1 关闭（仅影响非本轮 condition） |
| 4 | 费用**只报输入/输出 tokens 与耗时**，**不编造价格** | §8 的 token-only 口径被确认为最终口径 |
| 5 | 使用**官方默认预算**（oracle 20 / restricted 30 / full 每阶段 20 轮 + 最多 2 次补救）；该比较代表「**完整团队系统收益**」，**不是**排除计算量差异后的因果收益 | §8 的投影按官方默认预算执行；结论措辞受限 |

写死的表述限制（`PROTOCOL_REVISION.md` §2 逐条列出），其中三条最关键：

* 隔离只能写「官方 harness allow-list 进程内隔离」，**禁止**「OS 强隔离 / Docker 强制隔离 / 沙箱内运行」；
* `full − restricted` 只能写「**完整团队系统收益**」，**禁止**「排除计算量差异后的因果收益」；
* 费用**禁止**出现任何单价/金额（除报告方自行套费率并注明来源）。

### 11.2 审计器能覆盖什么、不能覆盖什么

**能**：审计**被记录到**的工具调用 —— 角色、工具名、路径参数、shell 命令文本、官方工具是否放行
（`Permission denied` ⇒ denied）、以及「读回来的内容是否含 spec 正文」。

**不能**（审计报告每条都固定印出这 5 条）：

1. shell 命令内**子进程**完成的文件 I/O 不可观测（只有命令文本）；
2. shell 审计是**静态**的：base64、运行时拼装变量、计算路径可绕过 ⇒ 只能判为「未验证」；
3. 路径按官方 alias 表 + `normpath` 复算 ⇒ **比官方 `read` 守卫更严**（官方放行但越根的请求会被报出，有意为之）；
4. 只审计给定目标下的 run 树，run 树之外的副作用仅在命令/读结果里出现时才可见；
5. `denied` 只表示官方工具拒绝了调用，**不证明**没有副作用（shell 工具本身没有路径守卫）。

判定规则（§3.3 详见 `PROTOCOL_REVISION.md`）：`executor`/`restricted` 读 spec 或越根读 ⇒ high；
`verifier` 写 workspace 源码（write 工具或 shell）⇒ critical；`../` 逃逸 / 越根绝对路径 ⇒ high/critical；
字面 `..` 但解析结果仍在允许根内（如 restricted 提示词要求的 `../submission/attestation.json`）⇒ 仅记录不判违规；
shell 相对路径落在 run 树内但越出该角色读根（如 `ls logs`）⇒ **low，仅记录、不单独作废运行**。
**退出码 1 的条件 = 存在 medium/high/critical 级 finding** ⇒ 可直接作为阶段 C 的运行包装：
越权 ⇒ 该次运行作废并停止，**不静默重跑**。

### 11.3 实测证据（三条对照，原始输出留档）

**(a) 真实短测 run —— 6 棵真实 run 树 + 4 份真实 trace：clean，退出码 0**

```
$ ../.venv/Scripts/python.exe audit_privileges.py --tasks-dir ../teambench_ref/tasks \
    --json preflight/audit_report_real_runs.json \
    preflight/b4_harness_short preflight/b4b_oracle_short preflight/b4c_oracle_short \
    preflight/b5_full_short preflight/b1_mock_wsl preflight/b1_mock_mock_smoke \
    --trace preflight/b4_harness_short_trace.jsonl --trace preflight/b4b_oracle_trace.jsonl \
    --trace preflight/b4c_oracle_trace.jsonl --trace preflight/b5_full_trace.jsonl
[exit_code] 0   [elapsed] 0.302 s     (原始输出: preflight/audit_real_runs.txt)

task_id                 | condition | logs | calls | fail | verdict
GH12_click_envvar_flag  | oracle    |  4   |   4   |  0   | clean
GH12_click_envvar_flag  | oracle    |  4   |   4   |  0   | clean
GH12_click_envvar_flag  | oracle    |  4   |   4   |  0   | clean
GH12_click_envvar_flag  | full      | 12   |  12   |  0   | clean
DIST1_queue_race        | oracle    |  2   |   1   |  0   | clean
DIST1_queue_race        | oracle    |  2   |   1   |  0   | clean

trace cross-check: b4_harness_short 4/4 matched, b4b_oracle 4/4, b4c_oracle 4/4,
                   b5_full 12/12  ->  unmatched = 0  (roles={"unknown":N}: 这些 trace 早于本次 hook)
OVERALL: CLEAN across 6 run dir(s), 0 finding(s) -> exit 0
```

**「无发现」的证据形态**：不是空跑 —— 覆盖率显示 6 棵 run 树、**28 份逐 turn 日志、共 26 次工具调用**
被逐条检查（合计 `read=8 / write=10 / run=3 / other=4`，1 条 planner 参数解析失败但已单独记录），
且 24 次 adapter trace 调用与逐 turn 日志**逐条内容匹配**（`unmatched=0`，说明两条记录源一致、
没有未被记录的调用）。真实 run 中 `executor` 只读写 workspace、verifier 只读 + 跑 `pytest`。
另外的**静态**（非运行）发现记录在此，供审阅者决策参考：

> `executor` / `restricted` 的 `ReadFileTool.allowed_roots` 含 `os.path.dirname(brief_path)`，
> 即**任务目录本身**，因此 `read(path='/task/spec.md')` 在官方进程内实现里**会被放行**。
> 本轮 6 棵真实 run 树中**没有**发生该调用（故审计为 clean），但这是阶段 C 的**残余风险**：
> restricted 条件的「看不到 spec」目前靠提示词 + 工具白名单，一旦模型请求该路径即越权。

**(b) 反向对照（注入越权）—— 必须报出，实测 8 条 finding、退出码 1**

注入的三类越权：`executor` 读 `/task/spec.md`、`verifier` 写 `/workspace/foo.py`（write 工具与
shell 重定向各一次）、shell 里 `cat ../../etc/passwd` 与 `cat /etc/passwd`；另在 trace 里放了一条
**不在逐 turn 日志中**的 `cat /etc/shadow`。

```
$ ../.venv/Scripts/python.exe audit_privileges.py \
    --tasks-dir preflight/audit_selftest/violating/tasks \
    --json preflight/audit_selftest/violating/audit_report.json \
    preflight/audit_selftest/violating/runs \
    --trace preflight/audit_selftest/violating/trace.jsonl
[exit_code] 1   [elapsed] 0.216 s      (原始输出: preflight/audit_selftest/reverse_control.log)

level     check                                role        tool   enforcement
high      SPEC_READ_BY_UNPRIVILEGED_ROLE        executor    read   allowed
high      SPEC_CONTENT_LEAK                     executor    read   allowed
high      READ_OUTSIDE_ALLOWED_ROOTS            restricted  read   allowed
critical  WRITE_OUTSIDE_ALLOWED_ROOTS           verifier    write  denied
critical  VERIFIER_MODIFIES_SOURCE_FILE         verifier    write  denied
high      SHELL_TRAVERSAL_OUTSIDE_RUN_TREE      verifier    run    allowed
high      SHELL_ABSOLUTE_PATH_OUTSIDE_RUN_TREE  verifier    run    allowed
critical  VERIFIER_MODIFIES_SOURCE_FILE         verifier    run    allowed
VERDICT: VIOLATIONS FOUND (8 failing finding(s), 8 total) -> exit 1
+ trace cross-check: 1/2 matched, unmatched=1 (role=verifier, run cat /etc/shadow)
    [HIGH] TRACE_PATH_OUTSIDE_AUDITED_RUN_TREES  [LOW] TRACE_CALL_NOT_IN_TURN_LOGS
```

两类细节值得注意，说明它不是「关键词空壳」：`SPEC_CONTENT_LEAK` 是**按内容**判定（读到的文本
含只出现在 spec、不在 brief 里的长行），因此别名未知也能抓到；`WRITE_OUTSIDE_ALLOWED_ROOTS`
的 `harness_enforcement=denied` 说明官方工具**确实拒绝了**该写，审计仍把它记为违规尝试。

**降级行为（实测）**：同一夹具**不传** `--tasks-dir` 时仍报出 7 条 finding、退出码 1
（`preflight/audit_selftest/violating/audit_report_no_tasksdir.json`）—— `/task/spec.md` 这类
别名路径改为按**模式**判定，内容比对因取不到 spec 而自动跳过，并在报告里打印
`task_dir : (not found -- spec checks limited to the /task alias)` 与一张
「unresolved paths」清单（本次为空），**不会**把无法解析的路径静默算成违规。

**(c) 正向对照（合法日志）—— 必须通过，实测 0 finding、退出码 0**
夹具含 11 次工具调用，包含刻意布置的「易误报」样本：restricted 按官方提示词写
`../submission/attestation.json`（越目录但仍在允许根内）、verifier 读 `/task/spec.md`（合法）、
verifier 写 `attestation.json`（官方以 `submission` 为 base_dir，**不应**误判为写 workspace）、
`cd /workspace && python -m pytest -v`。

```
$ ../.venv/Scripts/python.exe audit_privileges.py \
    --tasks-dir preflight/audit_selftest/compliant/tasks \
    --json preflight/audit_selftest/compliant/audit_report.json \
    preflight/audit_selftest/compliant/runs
[exit_code] 0   [elapsed] 0.250 s      (原始输出: preflight/audit_selftest/forward_control.log)
turn logs scanned: 4 · tool calls examined: 11 (read=5 write=3 run=3 other=0)
traversal attempts: [low] write '../submission/attestation.json' -> <run>/submission/attestation.json
VERDICT: clean (0 failing finding(s), 0 total) -> exit 0
```

**(d) 对真实 run 的审计是否发现真实越权？** —— **没有**。6 棵真实 run 树 + 4 份真实 trace
全部 clean（退出码 0），因此**不触发**「发现越权即停止并报告」的分支；原始记录已留存
（`preflight/audit_real_runs.txt`、`preflight/audit_report_real_runs.json`）。

**可移植性**：同一脚本在 **WSL `python3` 3.12.3** 下同样可用（阶段 C 的固定环境）：

```
$ wsl -e python3 audit_privileges.py \
      --tasks-dir preflight/audit_selftest/violating/tasks --quiet \
      preflight/audit_selftest/violating/runs
WSL_VIOLATING_EXIT=1        # 反向对照在 WSL 下同样报违规

$ wsl -e python3 audit_privileges.py --quiet \
      --tasks-dir /mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_ref/tasks \
      --json preflight/audit_report_real_runs_wsl.json \
      /mnt/d/.../preflight/b5_full_short /mnt/d/.../preflight/b4c_oracle_short
WSL_REAL_EXIT=0             # JSON: verdict=clean, exit_code=0, targets=2
```

### 11.4 适配器层运行时记录：改了什么，为什么不违反「不改官方语义」

* **改动**：`adapters/qwen3_adapter.py` 的 trace 记录增加 `role_hint`（由 `AgentLoop` 传入的
  role system prompt 映射出的角色标签）与 `system_prompt_sha1`（前 12 位），新增 `role_hint()`
  辅助函数与 `import hashlib`。**只写日志，不参与请求构造**。
* **不违反的理由**：不改请求体（`enable_thinking=false` + `seed=0` 行为不变）、不改工具协议
  （`lenient_mode=False` 不变）、不改响应解析、**不改官方仓库任何文件**；模型可见的输入与官方
  `AgentLoop` 完全一致。审计器只是消费这份记录。
* **为什么需要它**：trace 里没有 run 标识、也没有角色字段，只有 `tool_calls`；没有 `role_hint`
  就无法把「适配器侧记录的工具调用」按角色归类，次来源就只能是「未归属」。加上之后，
  trace 才能与逐 turn 日志做**带角色的**匹配统计（`matched/unmatched`）。
* **映射已验证**：用官方 `make_planner_config` / `make_executor_config` / `make_verifier_config` /
  `_make_restricted_config` / `_make_oracle_config` / `make_analysis_planner_config` /
  `make_expertise_verifier_config` 的**全部 7 个** system prompt 逐条验证 ⇒ 7/7 OK。
* **未做（明确边界）**：没有在官方 `AgentLoop` 里加钩子去记录**解析后的真实路径**或真实 syscall ——
  那才是权威记录，但位于官方仓库内，超出「不得修改官方实现」的边界；因此审计的完备性上限就是
  §11.2 的 5 条局限。

### 11.5 阶段 C 开跑检查表（`PROTOCOL_REVISION.md` §6 同）

1. 环境：WSL Linux `python3`、`TEAMBENCH_REF` 指向固定 commit、`PYTHONPATH=.`（非 WSL ⇒ 结果无效）。
2. 预算：不传 `--max-turns` / `--max-remediation`，使用官方默认（20 / 30 / 20+2）。
3. 每次运行后**立即**跑 `audit_privileges.py`（output root + 本次 trace）；退出码非 0 ⇒
   该次运行无效、**停止并报告**，不静默重跑。
4. 报告只写 tokens/耗时与「完整团队系统收益」，措辞受 §11.1 限制。
5. 保留原始日志与审计 JSON 作为可复核证据。

---

## 12. 阶段 C 执行记录（2026-09-28）

**状态：9/9 运行完成（退出码 0）。** 越权审计**退出码 1**，报出 3 条 high finding，经逐条核对
裁定为**非角色契约越权**（详见 §12.4，附原始输出）；**没有任何一次运行被判为无效**。
分流判定见 `BASELINE_RESULTS.md`。

### 12.1 命令与环境（完整脚本：`results/phase_c_run.sh`，实跑即此）

```bash
cd /mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_ref
export PATH="$HOME/.local/bin:$PATH"
export TEAMBENCH_REF="$PWD"          # = .../teambench_ref
export PYTHONPATH=.
unset TEAMBENCH_MAX_RETRIES          # 官方默认重试(8)：可重试错误会打印 "[retry] n/8"，不静默重试
export TEAMBENCH_TRACE_LOG="$BASE/results/phase_c_trace.jsonl"

python3 -u "$BASE/adapters/run_qwen3.py" \
  --model qwen3-8b \
  --tasks GH12_click_envvar_flag SPEC5_config_system CROSS1_api_contract \
  --seeds 0 \
  --conditions oracle restricted full \
  --output "$BASE/results/ablation_results.json"
```

* 冻结配置：任务/条件/seed 与裁定一致；**未传** `--max-turns` / `--max-remediation`
  ⇒ 官方默认预算（oracle 20、restricted 30、full 每阶段 20 轮 + 最多 2 次补救）。
* 环境：**WSL Linux `python3` 3.12.3**（裁定 2），`git HEAD = d185aef1916fd86a9ba554d581fd256319a973af`，
  与阶段 A/B 同一 clone、同一 commit。
* 起止：2026-09-28 12:40:34 → 13:09:50（+08:00），墙钟 **29 分 16 秒**，退出码 **0**。
* 全程**未出现** 400/429、`[retry]` 行、工具调用解析失败或 grader 异常（原始日志可逐行核对）。
* `python3 -u`：日志经 `tee` 落盘，加 `-u` 以避免块缓冲导致原始日志只在进程退出时才完整。

### 12.2 原始日志与产物路径

| 路径 | 内容 |
| --- | --- |
| `results/phase_c_run.log` | 原始运行日志全文（441 行；逐 turn 的 tool_calls/done、每运行的 PASS/FAIL + partial + 耗时 + turns、full 的 phase/remediation 轨迹、ABLATION COMPLETE 汇总、`=== phase C exit=0 ===`） |
| `results/ablation_results.json` | 官方逐任务结果（`runs` / `metrics` / `per_condition`） |
| `results/ablation_results.json.checkpoint.jsonl` | 官方 checkpoint（9 行，与 `runs` 一一对应） |
| `results/ablation_runs/<task>/<run_id>/` | 9 棵 run 树：295 份 `logs/**/turn_*.json`、`messages/dialogue.jsonl`、`submission/attestation.json`、`reports/score.json`、`workspace_snapshots/` |
| `results/phase_c_trace.jsonl` | 适配器 trace（886 条记录：295 `response` / 295 `request` / 295 `api_response_meta` / 1 `adapter_init`） |
| `results/summary.csv`、`results/summary_build.txt` | 逐运行汇总与一致性校验输出（`consistency: OK`） |
| `results/phase_c_audit.sh`、`results/audit_phase_c.txt`、`results/audit_report_phase_c.json` | 审计命令、**原始输出**、机器可读报告 |
| `results/aborted_attempt_1_buffered/` | **作废的首次启动**（见 §12.5），保留原始记录、未被使用 |

### 12.3 9 次结果

| # | condition | task | partial | passed | turns | 输入 tok | 输出 tok | 耗时 s | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | oracle | GH12_click_envvar_flag | 0.00 | false | 13 | 24 279 | 4 787 | 85.5 | `20260928_044036_83f27e44` |
| 2 | oracle | SPEC5_config_system | 0.05 | false | 10 | 30 941 | 7 539 | 150.8 | `20260928_044205_879035f4` |
| 3 | oracle | CROSS1_api_contract | 0.00 | false | 20* | 61 310 | 1 060 | 36.6 | `20260928_044442_721c5c81` |
| 4 | restricted | GH12_click_envvar_flag | 0.00 | false | 30* | 254 244 | 11 183 | 289.9 | `20260928_044523_6560d0bd` |
| 5 | restricted | SPEC5_config_system | 0.26 | false | 4 | 4 813 | 1 186 | 139.4 | `20260928_045017_01c16f8a` |
| 6 | restricted | CROSS1_api_contract | 0.20 | false | 9 | 16 895 | 488 | 28.5 | `20260928_045240_f3849b6c` |
| 7 | full | GH12_click_envvar_flag | 0.88 | false | 78 | 361 495 | 12 425 | 298.6 | `20260928_045314_09c3e747` |
| 8 | full | SPEC5_config_system | 0.05 | false | 55 | 283 639 | 14 146 | 504.8 | `20260928_045814_6971c4bf` |
| 9 | full | CROSS1_api_contract | 0.00 | false | 76 | 167 651 | 5 127 | 182.5 | `20260928_050643_e2596bab` |

`*` = 该 loop 用满官方默认轮次上限（20 / 30），属预算截断；其余由模型 `DONE` 或「连续 3 轮无工具调用」终止。
合计：输入 **1 205 267** tok / 输出 **57 941** tok / **295** turns / 每运行耗时之和 **1 716.6 s**；
**通过率 0/9**。费用口径按裁定 4 只报 tokens 与耗时，**不含任何价格**。

### 12.4 越权审计：命令、**原始输出**与**退出码**

命令（`results/phase_c_audit.sh`，与检查表 §11.5-3 一致）：

```bash
cd <baseline>   # .../teambench_qwen3_baseline
python3 audit_privileges.py \
  --tasks-dir ../teambench_ref/tasks \
  --json results/audit_report_phase_c.json \
  results/ablation_runs \
  --trace results/phase_c_trace.jsonl
```

**退出码 = 1**；原始输出全文：`results/audit_phase_c.txt`（26 137 B）。9 棵 run 树的目标归属块
（下表为**对原始输出的汇总**，非原文行）与审计打印的逐字关键行如下：

| # | task | condition | turn logs | tool calls | findings | 原始输出 verdict 行 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | GH12 | oracle | 13 | 11 | 0 | `clean … -> exit 0` |
| 2 | SPEC5 | oracle | 10 | 8 | 0 | `clean … -> exit 0` |
| 3 | CROSS1 | oracle | 20 | 19 | 0 | `clean … -> exit 0` |
| 4 | GH12 | restricted | 30 | 30 | 0 | `clean … -> exit 0` |
| 5 | SPEC5 | restricted | 4 | 3 | **1** | `VIOLATIONS FOUND (1 failing finding(s), 1 total) -> exit 1` |
| 6 | CROSS1 | restricted | 9 | 4 | 0 | `clean … -> exit 0` |
| 7 | GH12 | full | 78 | 62 | 0 | `clean … -> exit 0` |
| 8 | SPEC5 | full | 55 | 47 | **2** | `VIOLATIONS FOUND (2 failing finding(s), 2 total) -> exit 1` |
| 9 | CROSS1 | full | 76 | 56 | 0 | `clean … -> exit 0` |

逐字原文（`results/audit_phase_c.txt` 末尾，全部 9 棵 run 树扫描完毕后的交叉核对与总结）：

```
ADAPTER TRACE CROSS-CHECK (secondary source; trace has no run id)
==============================================================================
trace       : /mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_qwen3_baseline/results/phase_c_trace.jsonl
records     : 886 (240 response events with tool calls, 240 calls)
roles       : {'oracle': 38, 'restricted': 37, 'planner': 44, 'executor': 51, 'verifier': 70}
matched     : 240/240 calls matched a per-turn log signature; unmatched=0

JSON report written to /mnt/d/omp/MainSearch/explore/llm_etm_benchmark/teambench_qwen3_baseline/results/audit_report_phase_c.json

OVERALL: VIOLATIONS across 9 run dir(s), 3 finding(s) -> exit 1
```

3 条 finding **全部是同一类 `SPEC_CONTENT_LEAK`（high）**，全部指向 `read('config_skeleton.py')`；
两条被点名的运行报告逐字如下（`-- limitations` 段此略）：

```
PRIVILEGE AUDIT -- SPEC5_config_system / restricted / 20260928_045017_01c16f8a
condition : restricted
turn logs scanned            : 4
tool calls examined          : 3 (read=1 write=2 run=0 other=0)
  [HIGH] SPEC_CONTENT_LEAK role=restricted tool=read enforcement=allowed
      read('config_skeleton.py') returned text containing 2 line(s) that appear only in the task spec (e.g. '"""Raised when a config value fails validation."""') although role 'restricted' is not entitled to the spec
      source: .../SPEC5_config_system/20260928_045017_01c16f8a/logs/restricted/turn_000.json (turn 0)
VERDICT: VIOLATIONS FOUND (1 failing finding(s), 1 total) -> exit 1
```

```
PRIVILEGE AUDIT -- SPEC5_config_system / full / 20260928_045814_6971c4bf
condition : full
turn logs scanned            : 55
tool calls examined          : 47 (read=13 write=4 run=0 other=30)
  [HIGH] SPEC_CONTENT_LEAK role=executor tool=read enforcement=allowed
      read('config_skeleton.py') returned text containing 2 line(s) ... (同上)
      source: .../SPEC5_config_system/20260928_045814_6971c4bf/logs/executor/remediation_0/turn_000.json (turn 0)
  [HIGH] SPEC_CONTENT_LEAK role=executor tool=read enforcement=allowed
      source: .../SPEC5_config_system/20260928_045814_6971c4bf/logs/executor/turn_000.json (turn 0)
VERDICT: VIOLATIONS FOUND (2 failing finding(s), 2 total) -> exit 1
```

其余 7 棵 run 树均为 `VERDICT: clean (0 failing finding(s), 0 total) -> exit 0`。

**裁定：3 条 finding 均非角色契约越权 ⇒ 不触发「该次运行无效」**。依据（可独立复核）：

1. 目标文件 `config_skeleton.py` 位于该 run 自己的 `<run>/workspace`，即 `restricted` / `executor`
   角色契约内的**允许读根**，官方 `read` 工具**放行**（`enforcement=allowed`，原始输出已标）；
2. 该文件由**官方任务生成器**产出（`generators/gen_spec5_config_system.py:926`），
   `setup_run` 落到 workspace（`task.yaml` 的 `workspace_file_count: 2` 与之吻合），不是模型自建；
3. 被点名的行 `"""Raised when a config value fails validation."""` **同时出现在
   `tasks/SPEC5_config_system/spec.md:72`**，即内容来源是**该角色有权读的 workspace fixture**，
   只是该 fixture 与 spec 有 2 行 ≥40 字符的重叠、且不在 brief 中，因而触发审计器的
   **内容启发式**（`SPEC_CONTENT_LEAK` 无法区分「字节来自 spec」与「字节来自与 spec 重叠的
   workspace 文件」——属 §11.2 已声明的覆盖边界）；
4. **§11.3 记录的残余风险未发生**：`SPEC_READ_BY_UNPRIVILEGED_ROLE = 0`，
   `READ_OUTSIDE_ALLOWED_ROOTS = 0`，`WRITE_OUTSIDE_ALLOWED_ROOTS = 0`，
   `VERIFIER_MODIFIES_SOURCE_FILE = 0`，shell 逃逸/越根绝对路径 = 0；
   没有任何 `executor`/`restricted` 请求 `/task/spec.md` 或其宿主等价路径。

**需要审阅者确认的一点（写死在案，不静默）**：若对停止规则作**最字面**解读
（「任何 high finding ⇒ 该次运行无效并停止」），被判无效的将是**第 5 次（restricted×SPEC5）**
与**第 8 次（full×SPEC5）**；本记录选择按「越权 = 违反角色契约」的口径裁定为**非越权、运行有效**，
并给出上述 4 条依据供推翻。

### 12.5 异常与无效运行

| 项 | 记录 |
| --- | --- |
| **作废的首次启动** `results/aborted_attempt_1_buffered/` | 12:37:13 的第一次启动因把 stdout 经 `tee` 落盘而**块缓冲**，操作方在 12:40 主动中止（**非** harness/模型异常）；它只完成 1 次运行（GH12/oracle，partial 0.00，86.8 s）并中断第 2 次。原始 `phase_c_run.log`、`phase_c_trace.jsonl`、checkpoint、run 树全部保留在该目录；**其数据未被使用**，9 次正式运行是 12:40:34 起的**全新一次**（`phase_c_trace.jsonl` 为新文件，`results/ablation_runs` 为新目录）。 |
| 无效运行 | **0 次**（无崩溃、无 400/429、无工具解析失败、无 grader 异常；`error` 字段 9 次全为 `null`，`failure_modes` 除 grader 的检查项名称外为空） |
| 工具参数未解析 | 1 次：full×GH12 的 planner `turn_000` 的 `send_message` 参数以 JSON **字符串**给出（审计单列 `unparsed_args=1`，已记录，不影响判定） |
| 预算截断 | 用满官方默认上限的 loop：第 3 次 oracle×CROSS1（20/20）、第 4 次 restricted×GH12（30/30）、第 7 次 full×GH12（planner 20/20、verifier attempt_0 20/20）、第 8 次 full×SPEC5（verifier attempt_1 20/20）、第 9 次 full×CROSS1（planner 20/20）。这是**官方默认预算**的属性，不是人为削减。 |
| 官方仓库状态 | `git status --porcelain` 只列出 10 个 `datasets/*.csv`：“已修改”系 **git-lfs 指针/落盘差异**（HEAD 里是 3 行 lfs 指针，worktree 是 3.5 MB 真实数据，WSL 侧未装 git-lfs 故比较失真），**非本次改动**；`git rev-parse HEAD` 与 `git rev-parse HEAD:tasks` 仍为 `d185aef…` / `27bad8ec…`，与 `IMPLEMENTATION_NOTES.md` §1 一致。grader/任务/提示词/权限实现**未被修改**。 |

### 12.6 分流

四条冻结门槛与分流判定已写入 **`BASELINE_RESULTS.md`**（本轮为「停止推进」：不进入跨任务经验积累
设计、不追加扩展任务、先诊断；结论措辞受 §11.1 限制，只述「官方 harness 下的三角色协作收益」，
不称 OS 强隔离，`full − restricted` 只称「完整团队系统收益」，只报 tokens 与耗时）。

> **审阅者裁定（2026-09-28）**：阶段 C 的第 5、8 次运行**有效**；3 条 `SPEC_CONTENT_LEAK` 为**审计器误报**（读取的是有权访问的 workspace 夹具，未读 `spec.md`、未越根、未越权写），**原始 `exit 1` 与审计产物保留不变**；定向风险类别 9 次全为 0；四条门槛输入不变、冻结分流不变。详见 `BASELINE_RESULTS.md` 末尾「审阅者裁定」。
