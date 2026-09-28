"""
Configuration system skeleton for the application.

TODO: Implement this as config_system.py (not this file).

This skeleton shows the required interface. The Planner has the full
specification with all config keys, types, validation rules, and defaults.
"""
import json
import os
from typing import Any


# TODO: Implement ConfigValidationError
class ConfigValidationError(ValueError):
    """Raised when a config value fails validation."""
    pass


# TODO: Define the full schema here
# The schema maps config key names to their specification.
# This is a PARTIAL example — see the full spec for all keys.
# Example structure:
        # "host": ... (type: string, default: "0.0.0.0")
        # "port": ... (type: int, default: 6155)
        # "log_level": ... (type: enum, default: "WARN")
# ... (see spec for remaining keys)
_SCHEMA: dict[str, dict] = {
    # TODO: populate from spec
}


def load_config(
    config_file: str | None = None,
    env_vars: dict | None = None,
    cli_args: dict | None = None,
) -> dict:
    """
    Load and validate configuration from all sources.

    Priority (highest first): cli_args > env_vars > config_file > defaults

    TODO: Implement priority cascade
    TODO: Apply type coercion for each key
    TODO: Validate all values against the schema
    TODO: Apply defaults for missing keys
    """
    pass


def get_schema() -> dict:
    """Return the config schema."""
    # TODO: return _SCHEMA
    pass


def validate_value(key: str, value: Any) -> Any:
    """
    Validate and coerce a single value for the given config key.

    TODO: Look up key in _SCHEMA
    TODO: Coerce type (int, float, bool, enum, string)
    TODO: Validate range/allowed values
    TODO: Raise ConfigValidationError on failure
    """
    pass
