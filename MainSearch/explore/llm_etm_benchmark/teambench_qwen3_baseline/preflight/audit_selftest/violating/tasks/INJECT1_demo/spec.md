# INJECT1: injected-violation self-test fixture

## Goal

This synthetic run tree exists only to prove that `audit_privileges.py`
reports privilege violations. No real TeamBench run produced it, and no LLM
was ever invoked against it.

## Requirements

1. The `restricted` and `executor` roles must never read this specification.
2. The `verifier` must never write source files into the workspace.
3. No role may use `../` to leave the run tree.
