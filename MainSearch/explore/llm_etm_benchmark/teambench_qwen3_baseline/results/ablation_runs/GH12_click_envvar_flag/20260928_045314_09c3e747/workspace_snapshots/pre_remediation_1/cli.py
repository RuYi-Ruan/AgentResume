import os
import sys

def parse_bool_flag(flag_value, envvar_name: str, default: bool) -> bool:
    """Resolve a boolean flag from CLI arg, envvar, and default.

    Priority: explicit CLI arg > environment variable > default.

    BUG: the envvar branch uses bool(raw) which is True for ANY non-empty
    string, including "false", "0", "no".  Setting DEPLOY_DRY_RUN=false still
    activates --dry-run mode.
    """
    if flag_value is not None:
        # Explicit CLI flag always wins
        return bool(flag_value)

    raw = os.environ.get(envvar_name)
    if raw is not None:
        # Fix: explicitly check against known false values
        return raw.lower() not in ("0", "false", "no", "")

    return default


def deploy(dry_run_flag=None):
    """Deploy function that uses the parse_bool_flag function."""
    dry_run = parse_bool_flag(dry_run_flag, "DEPLOY_DRY_RUN", False)
    return {"dry_run": dry_run}
