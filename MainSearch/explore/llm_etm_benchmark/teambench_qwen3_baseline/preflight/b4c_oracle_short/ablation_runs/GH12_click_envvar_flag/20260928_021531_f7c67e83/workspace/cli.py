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
        # Fix: explicitly check for known false values
        return raw.lower() not in ("0", "false", "no", "")

    return default


def deploy(dry_run_flag=None):
    """Run the deployment operation.

    Args:
        dry_run_flag: explicit True/False from CLI, or None if not provided.

    Returns:
        dict with "dry_run" (bool) and "status" (str).
    """
    from config import load_config
    config = load_config()

    dry_run = parse_bool_flag(dry_run_flag, "DEPLOY_DRY_RUN", False)

    if dry_run:
        status = "dry_run: would deploy the application to {config['deploy_target']}">
    else:
        status = "executing: deploy the application to {config['deploy_target']}">

    return {"dry_run": dry_run, "status": status, "config": config}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="deploytool")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("deploy", help="deploy the application")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", dest="dry_run", action="store_true",
                   default=None, help="simulate deployment without making changes [DEPLOY_DRY_RUN]")
    g.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.set_defaults(**{"dry_run": None})

    args = parser.parse_args()
    if args.cmd == "deploy":
        result = deploy(getattr(args, "dry_run"))
        print(result["status"])
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()