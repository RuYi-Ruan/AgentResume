"""Tests for deploytool boolean flag / envvar interaction.

The core requirement: parse_bool_flag() must correctly interpret envvar
strings.  "false", "0", "no" must yield False; "true", "1", "yes" must
yield True; absent envvar falls back to the default.
"""
import os
import importlib
import pytest


def _clean_env():
    """Remove the test envvar so each test starts clean."""
    os.environ.pop("DEPLOY_DRY_RUN", None)


def _reload():
    import cli as m
    importlib.reload(m)
    return m


def test_envvar_false_string_gives_false():
    """DEPLOY_DRY_RUN=false must produce dry_run=False (core bug)."""
    _clean_env()
    os.environ["DEPLOY_DRY_RUN"] = "false"
    m = _reload()
    result = m.deploy(dry_run_flag=None)
    assert result["dry_run"] is False, (
        f"DEPLOY_DRY_RUN=false produced dry_run={result['dry_run']}, expected False"
    )
    _clean_env()


def test_envvar_zero_string_gives_false():
    """DEPLOY_DRY_RUN=0 must produce dry_run=False."""
    _clean_env()
    os.environ["DEPLOY_DRY_RUN"] = "0"
    m = _reload()
    result = m.deploy(dry_run_flag=None)
    assert result["dry_run"] is False, (
        f"DEPLOY_DRY_RUN=0 produced dry_run={result['dry_run']}, expected False"
    )
    _clean_env()


def test_envvar_no_string_gives_false():
    """DEPLOY_DRY_RUN=no must produce dry_run=False."""
    _clean_env()
    os.environ["DEPLOY_DRY_RUN"] = "no"
    m = _reload()
    result = m.deploy(dry_run_flag=None)
    assert result["dry_run"] is False, (
        f"DEPLOY_DRY_RUN=no produced dry_run={result['dry_run']}, expected False"
    )
    _clean_env()


def test_envvar_true_string_gives_true():
    """DEPLOY_DRY_RUN=true must produce dry_run=True."""
    _clean_env()
    os.environ["DEPLOY_DRY_RUN"] = "true"
    m = _reload()
    result = m.deploy(dry_run_flag=None)
    assert result["dry_run"] is True, (
        f"DEPLOY_DRY_RUN=true produced dry_run={result['dry_run']}, expected True"
    )
    _clean_env()


def test_envvar_one_string_gives_true():
    """DEPLOY_DRY_RUN=1 must produce dry_run=True."""
    _clean_env()
    os.environ["DEPLOY_DRY_RUN"] = "1"
    m = _reload()
    result = m.deploy(dry_run_flag=None)
    assert result["dry_run"] is True, (
        f"DEPLOY_DRY_RUN=1 produced dry_run={result['dry_run']}, expected True"
    )
    _clean_env()


def test_no_envvar_uses_default():
    """Without envvar, dry_run must equal the hardcoded default (False)."""
    _clean_env()
    m = _reload()
    result = m.deploy(dry_run_flag=None)
    assert result["dry_run"] is False, (
        f"No envvar produced dry_run={result['dry_run']}, expected False"
    )
    _clean_env()


def test_explicit_true_wins_over_false_envvar():
    """Explicit flag=True overrides DEPLOY_DRY_RUN=false."""
    _clean_env()
    os.environ["DEPLOY_DRY_RUN"] = "false"
    m = _reload()
    result = m.deploy(dry_run_flag=True)
    assert result["dry_run"] is True
    _clean_env()


def test_explicit_false_wins_over_true_envvar():
    """Explicit flag=False overrides DEPLOY_DRY_RUN=true."""
    _clean_env()
    os.environ["DEPLOY_DRY_RUN"] = "true"
    m = _reload()
    result = m.deploy(dry_run_flag=False)
    assert result["dry_run"] is False
    _clean_env()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
