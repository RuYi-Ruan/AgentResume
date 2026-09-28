from typing import Any, Dict, Optional
import os
import json
from enum import Enum

# Custom exception for configuration validation errors
class ConfigValidationError(ValueError):
    """Raised when a config value fails validation."""
    pass

# Define the full schema based on the specification
_SCHEMA: dict[str, dict] = {
    "queue_url": {
        "type": "string",
        "default": "redis://localhost:6379/0",
        "env_var": "CELERY_QUEUE_URL",
        "validation": {"non-empty string": True},
        "description": "URL of the message queue"
    },
    "concurrency": {
        "type": "int",
        "default": 3,
        "env_var": "CELERY_CONCURRENCY",
        "validation": {"range": [1, 32]},
        "description": "Number of concurrent workers; must be 1-32"
    },
    "max_retries": {
        "type": "int",
        "default": 8,
        "env_var": "CELERY_MAX_RETRIES",
        "validation": {"range": [0, 20]},
        "description": "Maximum retry attempts per job; must be 0-20"
    },
    "retry_backoff_seconds": {
        "type": "int",
        "default": 1,
        "env_var": "CELERY_RETRY_BACKOFF",
        "validation": {"range": [1, 300]},
        "description": "Seconds to wait between retries; must be 1-300"
    },
    "job_timeout": {
        "type": "int",
        "default": 300,
        "env_var": "CELERY_JOB_TIMEOUT",
        "validation": {"range": [1, 3600]},
        "description": "Job execution timeout in seconds; must be 1-3600"
    },
    "log_level": {
        "type": "enum",
        "default": "INFO",
        "env_var": "CELERY_LOG_LEVEL",
        "validation": {"allowed": ["DEBUG", "INFO", "WARN"]},
        "description": "Logging verbosity; one of ['DEBUG', 'INFO', 'WARN']"
    },
    "dead_letter_queue": {
        "type": "bool",
        "default": True,
        "env_var": "CELERY_DEAD_LETTER",
        "validation": {"bool": True},
        "description": "Route failed jobs to dead letter queue"
    },
    "heartbeat_interval": {
        "type": "int",
        "default": 60,
        "env_var": "CELERY_HEARTBEAT",
        "validation": {"range": [5, 300]},
        "description": "Worker heartbeat interval seconds; must be 5-300"
    },
    "prefetch_count": {
        "type": "int",
        "default": 10,
        "env_var": "CELERY_PREFETCH",
        "validation": {"range": [1, 100]},
        "description": "Number of messages to prefetch; must be 1-100"
    },
    "ack_on_failure": {
        "type": "bool",
        "default": False,
        "env_var": "CELERY_ACK_ON_FAILURE",
        "validation": {"bool": True},
        "description": "Acknowledge message even on job failure"
    },
    "metrics_enabled": {
        "type": "bool",
        "default": True,
        "env_var": "CELERY_METRICS",
        "validation": {"bool": True},
        "description": "Enable Prometheus metrics"
    }
}

def load_config(
    config_file: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
    cli_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load and validate configuration from all sources in priority order.

    Args:
        config_file: Path to a JSON config file (optional).
        env_vars: Dict of environment variables (defaults to os.environ).
        cli_args: Dict of CLI arguments — highest priority.

    Returns:
        A dict with all config keys populated, validated, and type-coerced.

    Raises:
        ConfigValidationError: If any value fails validation.
        FileNotFoundError: If config_file is specified but does not exist.
    """
    # Initialize config with defaults
    config = {key: value["default"] for key, value in _SCHEMA.items()}

    # Apply CLI arguments first (highest priority)
    if cli_args:
        for key, value in cli_args.items():
            if key in config:
                config[key] = value

    # Apply environment variables
    if env_vars is None:
        env_vars = os.environ
    for key, value in env_vars.items():
        if key in _SCHEMA:
            config[key] = value

    # Apply config file (lowest priority)
    if config_file:
        try:
            with open(config_file, "r") as f:
                file_config = json.load(f)
            for key, value in file_config.items():
                if key in config:
                    config[key] = value
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in config file: {config_file}")

    # Validate and coerce each value
    for key, value in config.items():
        config[key] = validate_value(key, value)

    return config


def get_schema() -> dict:
    """Return the config schema."""
    return _SCHEMA


def validate_value(key: str, value: Any) -> Any:
    """Validate and coerce a single value against the schema for `key`.

    Returns the coerced value.
    Raises ConfigValidationError if invalid.
    """
    schema = _SCHEMA.get(key)
    if not schema:
        raise ConfigValidationError(f"Unknown config key: {key}")

    # Coerce type
    if schema["type"] == "string":
        coerced_value = str(value)
    elif schema["type"] == "int":
        try:
            coerced_value = int(value)
        except (TypeError, ValueError):
            raise ConfigValidationError(f"Value for {key} is not an integer: {value}")
    elif schema["type"] == "float":
        try:
            coerced_value = float(value)
        except (TypeError, ValueError):
            raise ConfigValidationError(f"Value for {key} is not a float: {value}")
    elif schema["type"] == "bool":
        if isinstance(value, bool):
            coerced_value = value
        elif isinstance(value, str):
            value = value.lower()
            if value in ["true", "yes", "on", "1"]:
                coerced_value = True
            elif value in ["false", "no", "off", "0"]:
                coerced_value = False
            else:
                raise ConfigValidationError(f"Value for {key} is not a boolean: {value}")
        else:
            raise ConfigValidationError(f"Value for {key} is not a boolean: {value}")
    elif schema["type"] == "enum":
        allowed_values = schema["validation"]("allowed")
        if isinstance(value, str):
            value = value.strip().upper()
        if value not in allowed_values:
            raise ConfigValidationError(f"Value for {key} is not allowed: {value}")
        coerced_value = value
    else:
        raise ConfigValidationError(f"Unsupported type for {key}: {schema["type"]}")

    # Validate range or allowed values
    if schema["type"] == "int":
        min_val, max_val = schema["validation"]("range")
        if not (min_val <= coerced_value <= max_val):
            raise ConfigValidationError(f"Value for {key} is out of range [{min_val}, {max_val}]: {coerced_value}")
    elif schema["type"] == "enum":
        allowed_values = schema["validation"]("allowed")
        if coerced_value not in allowed_values:
            raise ConfigValidationError(f"Value for {key} is not allowed: {coerced_value}")
    elif schema["type"] == "bool":
        # No additional validation needed for bool
        pass
    else:
        # No additional validation for other types
        pass

    return coerced_value