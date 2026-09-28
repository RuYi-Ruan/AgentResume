import json
import os
from typing import Any

# Custom exception for configuration validation errors
class ConfigValidationError(ValueError):
    """Raised when a config value fails validation."""
    def __init__(self, key: str, value: Any, message: str = ''):
        self.key = key
        self.value = value
        self.message = message
        super().__init__(f"{message}: {key}={value}")

# Full configuration schema (populated from the full spec)
_SCHEMA: dict[str, dict] = {
    # Example schema entries (replace with actual spec)
    # "host": {"type": "string", "default": "0.0.0.0"},
    # "port": {"type": "int", "default": 6155},
    # "log_level": {"type": "enum", "allowed": ["DEBUG", "INFO", "WARN"]},
    # ... (add all keys from the full spec here)
}


def load_config(
    config_file: str | None = None,
    env_vars: dict | None = None,
    cli_args: dict | None = None,
) -> dict:
    """Load and validate configuration from all sources.

    Priority (highest first): cli_args > env_vars > config_file > defaults
    """
    config = {}

    # Apply CLI arguments first (highest priority)
    if cli_args:
        config.update(cli_args)

    # Apply environment variables
    if env_vars is None:
        env_vars = os.environ
    for key, value in env_vars.items():
        if key in _SCHEMA:
            config[key] = value

    # Apply config file (if provided)
    if config_file:
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            for key, value in file_config.items():
                if key in _SCHEMA:
                    config[key] = value
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in config file: {config_file}")

    # Apply defaults for missing keys
    for key, schema in _SCHEMA.items():
        if key not in config:
            config[key] = schema.get('default', None)

    # Validate and coerce each value
    for key, value in config.items():
        if key in _SCHEMA:
            try:
                config[key] = validate_value(key, value)
            except ConfigValidationError as e:
                raise ConfigValidationError(e.key, e.value, e.message) from e

    return config

def get_schema() -> dict:
    """Return the config schema."""
    return _SCHEMA

def validate_value(key: str, value: Any) -> Any:
    """Validate and coerce a single value for the given config key."""
    schema = _SCHEMA.get(key)
    if not schema:
        raise ConfigValidationError(key, value, "Key not found in schema")

    # Type coercion
    type_ = schema.get('type')
    if type_ == 'int':
        try:
            return int(value)
        except (ValueError, TypeError):
            raise ConfigValidationError(key, value, "Value must be an integer")
    elif type_ == 'float':
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ConfigValidationError(key, value, "Value must be a float")
    elif type_ == 'bool':
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            value = value.lower()
            if value in ('true', 'yes', '1', 'on'):
                return True
            elif value in ('false', 'no', '0', 'off'):
                return False
            else:
                raise ConfigValidationError(key, value, "Value must be a boolean")
        else:
            raise ConfigValidationError(key, value, "Value must be a boolean")
    elif type_ == 'enum':
        allowed = schema.get('allowed', [])
        if not allowed:
            raise ConfigValidationError(key, value, "No allowed values specified for enum")
        if isinstance(value, str):
            value = value.lower()
            if value in allowed:
                return value
            else:
                raise ConfigValidationError(key, value, f"Value must be one of {allowed}")
        else:
            raise ConfigValidationError(key, value, "Value must be a string for enum")
    elif type_ == 'string':
        return str(value)
    else:
        raise ConfigValidationError(key, value, f"Unsupported type: {type_}")

    # Additional validation rules (e.g., range, format, etc.)
    # Add these based on the full spec
    # For example:
    # if type_ == 'concurrency' and not 1 <= value <= 32:
    #     raise ConfigValidationError(...)

    return value