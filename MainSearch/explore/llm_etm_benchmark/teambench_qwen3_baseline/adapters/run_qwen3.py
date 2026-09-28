"""Run the official TeamBench ablation with the qwen3-8b adapter registered.

Why a wrapper instead of editing the harness
--------------------------------------------
`harness/adapters/__init__.py::create_adapter` routes models by name prefix and
has no route for a third-party OpenAI-compatible endpoint. The official
`example_custom_adapter.py` suggests *editing* that factory; we deliberately do
not, so that `git -C teambench_ref status` stays clean (required evidence for the
review package). Instead this wrapper installs the extra route at runtime, before
the harness resolves the adapter, and then defers to the official CLI.

`run_full_ablation` imports `create_adapter` lazily inside the function body, so
patching `harness.adapters.create_adapter` before calling `main()` is sufficient
and affects every role (oracle/restricted/planner/executor/verifier) equally.

Usage (identical arguments to `python -m harness.ablation`):
    python run_qwen3.py --model qwen3-8b --tasks GH12_click_envvar_flag \
        --seeds 0 --conditions oracle --output results/ablation_results.json
"""

from __future__ import annotations

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path(
    os.environ.get("TEAMBENCH_REF", "D:/omp/MainSearch/explore/llm_etm_benchmark/teambench_ref")
)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import harness.adapters as _adapters  # noqa: E402
from qwen3_adapter import create_qwen3_adapter  # noqa: E402

_ORIGINAL_CREATE_ADAPTER = _adapters.create_adapter


def create_adapter(model: str, temperature: float = 0.2, **kwargs):
    """Route `qwen3*` model ids to the third-party endpoint; everything else unchanged."""
    if str(model).lower().startswith("qwen3"):
        return create_qwen3_adapter(model=model, temperature=temperature, **kwargs)
    return _ORIGINAL_CREATE_ADAPTER(model=model, temperature=temperature, **kwargs)


_adapters.create_adapter = create_adapter


def main() -> None:
    from harness import ablation

    ablation.main()


if __name__ == "__main__":
    main()
