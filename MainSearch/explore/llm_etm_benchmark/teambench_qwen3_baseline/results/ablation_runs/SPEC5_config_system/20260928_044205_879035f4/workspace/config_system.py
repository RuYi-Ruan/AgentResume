import json
import os
from typing import Any, Dict, Optional

# Custom exception for configuration validation errors
class ConfigValidationError(ValueError):
    """Raised when a config value fails validation."""
    def __init__(self, key: str, value: Any, message: str):
        self.key = key
        self.value = value
        self.message = message
        super().__init__(f"{key}: {message} (value: {value})")

# Full schema based on the specification
_SCHEMA: dict[str, dict] = {
    "queue_url": {
        "type": "string",
        "default": "redis://localhost:6379/0",
        "env_var": "CELERY_QUEUE_URL",
        "validation": {"non-empty string"},
        "description": "URL of the message queue"
    },
    "concurrency": {
        "type": "int",
        "default": 3,
        "env_var": "CELERY_CONCURRENCY",
        "validation": {"int in range [1, 32]"},
        "description": "Number of concurrent workers; must be 1-32"
    },
    "max_retries": {
        "type": "int",
        "default": 8,
        "env_var": "CELERY_MAX_RETRIES",
        "validation": {"int in range [0, 20]"},
        "description": "Maximum retry attempts per job; must be 0-20"
    },
    "retry_backoff_seconds": {
        "type": "int",
        "default": 1,
        "env_var": "CELERY_RETRY_BACKOFF",
        "validation": {"int in range [1, 300]"},
        "description": "Seconds to wait between retries; must be 1-300"
    },
    "job_timeout": {
        "type": "int",
        "default": 300,
        "env_var": "CELERY_JOB_TIMEOUT",
        "validation": {"int in range [1, 3600]"},
        "description": "Job execution timeout in seconds; must be 1-3600"
    },
    "log_level": {
        "type": "enum",
        "default": "INFO",
        "env_var": "CELERY_LOG_LEVEL",
        "validation": {"one of ['DEBUG', 'INFO', 'WARN']"},
        "description": "Logging verbosity; one of ['DEBUG', 'INFO', "WARN"]"
    },
    "dead_letter_queue": {
        "type": "bool",
        "default": True,
        "env_var": "CELERY_DEAD_LETTER",
        "validation": {"bool"},
        "description": "Route failed jobs to dead letter queue"
    },
    "heartbeat_interval": {
        "type": "int",
        "default": 60,
        "env_var": "CELERY_HEARTBEAT",
        "validation": {"int in range [5, 300]"},
        "description": "Worker heartbeat interval seconds; must be 5-300"
    },
    "prefetch_count": {
        "type": "int",
        "default": 10,
        "env_var": "CELERY_PREFETCH",
        "validation": {"int in range [1, 100]"},
        "description": "Number of messages to prefetch; must be 1-100"
    },
    "ack_on_failure": {
        "type": "bool",
        "default": False,
        "env_var": "CELERY_ACK_ON_FAILURE",
        "validation": {"bool"},
        "description": "Acknowledge message even on job failure"
    },
    "metrics_enabled": {
        "type": "bool",
        "default": True,
        "env_var": "CELERY_METRICS",
        "validation": {"bool"},
        "description": "Enable Prometheus metrics"
    }
}

# Load and validate configuration from all sources
def load_config(
    config_file: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
    cli_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load and validate configuration from all sources in priority order.

    Priority (highest first): cli_args > env_vars > config_file > defaults

    Args:
        config_file: Path to a JSON config file (optional).
        env_vars: Dict of environment variables (defaults to os.environ if None).
        cli_args: Dict of CLI arguments — highest priority.

    Returns:
        A dict with all config keys populated, validated, and type-coerced.

    Raises:
        ConfigValidationError: If any value fails validation.
        FileNotFoundError: If config_file is specified but does not exist.
    """
    # Initialize config with defaults
    config = {key: spec["default"] for key, spec in _SCHEMA.items()}

    # Apply CLI arguments (highest priority)
    if cli_args:
        for key, value in cli_args.items():
            if key in config:
                config[key] = value

    # Apply environment variables
    if env_vars is None:
        env_vars = os.environ
    for key, spec in _SCHEMA.items():
        env_var = spec.get("env_var")
        if env_var and env_var in env_vars:
            config[key] = env_vars[env_var]

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
            raise ValueError(f"Invalid JSON format in config file: {config_file}")

    # Validate and coerce each value
    for key, value in config.items():
        try:
            config[key] = validate_value(key, value)
        except ConfigValidationError as e:
            raise ConfigValidationError(e.key, e.value, e.message) from e

    return config

# Return the full schema
def get_schema() -> dict:
    """Return the config schema."""
    return _SCHEMA

# Validate and coerce a single value for the given config key
def validate_value(key: str, value: Any) -> Any:
    """
    Validate and coerce a single value for the given config key.

    Args:
        key: The config key to validate.
        value: The value to validate and coerce.

    Returns:
        The coerced value.

    Raises:
        ConfigValidationError: If the value fails validation.
    """
    spec = _SCHEMA.get(key)
    if not spec:
        raise ConfigValidationError(key, value, "Unknown config key")

    # Coerce type
    if spec["type"] == "int":
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise ConfigValidationError(key, value, "Value must be an integer")
    elif spec["type"] == "float":
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ConfigValidationError(key, value, "Value must be a float")
    elif spec["type"] == "bool":
        # Accept true/false (case-insensitive), 1/0, yes/no, on/off
        value_str = str(value).lower()
        if value_str in ["true", "yes", "on", "1"]:
            value = True
        elif value_str in ["false", "no", "off", "0"]:
            value = False
        else:
            raise ConfigValidationError(key, value, "Value must be a boolean (true/false, 1/0, yes/no, on/off)")
    elif spec["type"] == "enum":
        allowed_values = spec.get("validation", []).split(" ")
        if value not in allowed_values:
            raise ConfigValidationError(key, value, f"Value must be one of {allowed_values}")
    elif spec["type"] == "string":
        pass  # Use as-is
    else:
        raise ConfigValidationError(key, value, f"Unsupported type: {spec["type"]}")

    # Validate range or allowed values
    if spec["type"] == "int":
        min_val = spec.get("validation", {}).get("min", None)
        max_val = spec.get("validation", {}).get("max", None)
        if min_val is not None and value < min_val:
            raise ConfigValidationError(key, value, f"Value must be at least {min_val}")
        if max_val is not None and value > max_val:
            raise ConfigValidationError(key, value, f"Value must be at most {max_val}")
    elif spec["type"] == "enum":
        allowed_values = spec.get("validation", []).split(" ")
        if value not in allowed_values:
            raise ConfigValidationError(key, value, f"Value must be one of {allowed_values}")
    elif spec["type"] == "bool":
        pass  # Already validated above
    elif spec["type"] == "string":
        if spec.get("validation") and not value:
            raise ConfigValidationError(key, value, "Value must be non-empty string")
    else:
        pass  # No additional validation needed

    return value