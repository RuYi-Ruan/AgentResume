import json
import os
from typing import Any, Dict, Optional

class ConfigValidationError(ValueError):
    """Raised when a config value fails validation."""
    def __init__(self, key: str, value: Any, message: str):
        self.key = key
        self.value = value
        self.message = message
        super().__init__(f"{key}: {message} (value: {value})")

# Full configuration schema with all keys, types, validation rules, and defaults
_SCHEMA: dict[str, dict] = {
    "host": {
        "type": "string",
        "default": "0.0.0.0",
        "env_var": "WEB_HOST",
        "description": "Hostname or IP address to bind"
    },
    "port": {
        "type": "int",
        "default": 6155,
        "env_var": "WEB_PORT",
        "description": "TCP port to listen on; must be 2048-49151"
    },
    "log_level": {
        "type": "enum",
        "default": "WARN",
        "env_var": "WEB_LOG_LEVEL",
        "description": "Logging verbosity; one of ['INFO', 'WARN', "ERROR"]"
    },
    "request_timeout": {
        "type": "int",
        "default": 120,
        "env_var": "WEB_REQUEST_TIMEOUT",
        "description": "Request timeout in seconds; must be 1-3600"
    },
    "max_connections": {
        "type": "int",
        "default": 348,
        "env_var": "WEB_MAX_CONNECTIONS",
        "description": "Maximum concurrent connections; must be 1-1000"
    }
}

def load_config(
    config_file: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
    cli_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load and validate configuration from all sources.

    Priority (highest first): cli_args > env_vars > config_file > defaults
    """
    config = {}

    # Apply CLI arguments first (highest priority)
    if cli_args:
        config.update(cli_args)

    # Apply environment variables
    if env_vars:
        for key, value in env_vars.items():
            config[key] = value
    else:
        # Use os.environ if env_vars is not provided
        for key, value in os.environ.items():
            if key.startswith("WEB_"):
                config[key[4:]] = value

    # Apply config file (lowest priority)
    if config_file:
        try:
            with open(config_file, "r") as f:
                file_config = json.load(f)
            # Merge config file values into the config dictionary
            for key, value in file_config.items():
                if key not in config:
                    config[key] = value
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in config file: {config_file}")

    # Apply defaults for missing keys
    for key, spec in _SCHEMA.items():
        if key not in config:
            config[key] = spec.get("default")

    # Validate and coerce each value
    for key, value in config.items():
        try:
            config[key] = validate_value(key, value)
        except ConfigValidationError as e:
            raise RuntimeError(f"Validation failed for key '{key}': {e}") from e

    return config

def get_schema() -> dict:
    """Return the config schema."""
    return _SCHEMA

def validate_value(key: str, value: Any) -> Any:
    """Validate and coerce a single value for the given config key."""
    spec = _SCHEMA.get(key)
    if not spec:
        raise ConfigValidationError(key, value, "Key not found in schema")

    # Type coercion
    if spec["type"] == "int":
        try:
            return int(value)
        except (ValueError, TypeError):
            raise ConfigValidationError(key, value, "Value must be an integer")
    elif spec["type"] == "float":
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ConfigValidationError(key, value, "Value must be a float")
    elif spec["type"] == "bool":
        # Accept true/false, 1/0, yes/no, on/off (case-insensitive)
        value_str = str(value).lower()
        if value_str in ["true", "yes", "on", "1"]:
            return True
        elif value_str in ["false", "no", "off", "0"]:
            return False
        else:
            raise ConfigValidationError(key, value, "Value must be a boolean (true/false, 1/0, yes/no, on/off)")
    elif spec["type"] == "enum":
        allowed_values = spec.get("allowed_values", [])
        if not allowed_values:
            raise ConfigValidationError(key, value, "No allowed values specified for enum")
        value_str = str(value).lower()
        if value_str in [val.lower() for val in allowed_values]:
            return value_str
        else:
            raise ConfigValidationError(key, value, f"Value must be one of {allowed_values}")
    elif spec["type"] == "string":
        return str(value)
    else:
        raise ConfigValidationError(key, value, f"Unsupported type: {spec["type"]}")

    # Additional validation for specific keys
    if key == "port":
        if not (2048 <= config[key] <= 49151):
            raise ConfigValidationError(key, value, "Port must be between 2048 and 49151")
    elif key == "request_timeout":
        if not (1 <= config[key] <= 3600):
            raise ConfigValidationError(key, value, "Request timeout must be between 1 and 3600 seconds")
    elif key == "max_connections":
        if not (1 <= config[key] <= 1000):
            raise ConfigValidationError(key, value, "Max connections must be between 1 and 1000")
    elif key == "log_level":
        if str(value).lower() not in ["info", "warn", "error"]:
            raise ConfigValidationError(key, value, "Log level must be one of 'INFO', 'WARN', or 'ERROR'")

    return value