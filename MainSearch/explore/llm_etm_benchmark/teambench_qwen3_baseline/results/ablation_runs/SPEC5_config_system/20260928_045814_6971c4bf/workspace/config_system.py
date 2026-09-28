from typing import Any, Dict, Optional
import os
import json

class ConfigValidationError(ValueError):
    """Custom exception for configuration validation errors."""

    def __init__(self, key: str, value: Any, message: str):
        self.key = key
        self.value = value
        self.message = message
        super().__init__(f"{message}: {key}={value}")

_SCHEMA: Dict[str, Dict] = {
    "concurrency": {
        "type": "int",
        "default": 8,
        "description": "Number of worker threads to run in parallel.",
        "enum": [1, 2, 4, 8, 16, 32],
        "env_var": "WORKER_CONCURRENCY"
    },
    "log_level": {
        "type": "str",
        "default": "INFO",
        "description": "Log level for the application (DEBUG/INFO/WARN).",
        "enum": ["DEBUG", "INFO", "WARN"],
        "env_var": "LOG_LEVEL"
    },
    "dead_letter_queue": {
        "type": "bool",
        "default": False,
        "description": "Enable dead letter queue for failed messages.",
        "env_var": "DEAD_LETTER_QUEUE"
    },
    "max_retries": {
        "type": "int",
        "default": 3,
        "description": "Maximum number of retries for failed messages.",
        "env_var": "MAX_RETRIES"
    },
    "timeout": {
        "type": "float",
        "default": 5.0,
        "description": "Timeout for processing messages in seconds.",
        "env_var": "TIMEOUT"
    },
    "queue_url": {
        "type": "str",
        "default": "https://example.com/queue",
        "description": "URL of the message queue service.",
        "env_var": "QUEUE_URL"
    },
    "aws_access_key_id": {
        "type": "str",
        "default": "",
        "description": "AWS access key ID for authentication.",
        "env_var": "AWS_ACCESS_KEY_ID"
    },
    "aws_secret_access_key": {
        "type": "str",
        "default": "",
        "description": "AWS secret access key for authentication.",
        "env_var": "AWS_SECRET_ACCESS_KEY"
    },
    "aws_region": {
        "type": "str",
        "default": "us-east-1",
        "description": "AWS region for the service.",
        "env_var": "AWS_REGION"
    },
    "enable_metrics": {
        "type": "bool",
        "default": False,
        "description": "Enable metrics collection for performance monitoring.",
        "env_var": "ENABLE_METRICS"
    },
    "metrics_interval": {
        "type": "int",
        "default": 60,
        "description": "Interval in seconds for metrics collection.",
        "env_var": "METRICS_INTERVAL"
    },
    "health_check_interval": {
        "type": "int",
        "default": 300,
        "description": "Interval in seconds for health checks.",
        "env_var": "HEALTH_CHECK_INTERVAL"
    },
    "health_check_timeout": {
        "type": "int",
        "default": 10,
        "description": "Timeout in seconds for health checks.",
        "env_var": "HEALTH_CHECK_TIMEOUT"
    },
    "worker_id": {
        "type": "str",
        "default": "worker-01",
        "description": "Unique identifier for the worker instance.",
        "env_var": "WORKER_ID"
    },
    "max_message_size": {
        "type": "int",
        "default": 1024,
        "description": "Maximum size of a single message in bytes.",
        "env_var": "MAX_MESSAGE_SIZE"
    },
    "max_batch_size": {
        "type": "int",
        "default": 100,
        "description": "Maximum number of messages processed in a single batch.",
        "env_var": "MAX_BATCH_SIZE"
    },
    "auto_scaling": {
        "type": "bool",
        "default": False,
        "description": "Enable auto-scaling based on workload.",
        "env_var": "AUTO_SCALING"
    },
    "scale_up_threshold": {
        "type": "int",
        "default": 80,
        "description": "Percentage threshold for scaling up workers.",
        "env_var": "SCALE_UP_THRESHOLD"
    },
    "scale_down_threshold": {
        "type": "int",
        "default": 20,
        "description": "Percentage threshold for scaling down workers.",
        "env_var": "SCALE_DOWN_THRESHOLD"
    },
    "max_workers": {
        "type": "int",
        "default": 10,
        "description": "Maximum number of workers allowed in the system.",
        "env_var": "MAX_WORKERS"
    },
    "min_workers": {
        "type": "int",
        "default": 2,
        "description": "Minimum number of workers always running.",
        "env_var": "MIN_WORKERS"
    },
    "keep_alive_timeout": {
        "type": "int",
        "default": 300,
        "description": "Timeout in seconds before idle workers are terminated.",
        "env_var": "KEEP_ALIVE_TIMEOUT"
    },
    "retry_backoff": {
        "type": "int",
        "default": 5,
        "description": "Backoff time in seconds between retries.",
        "env_var": "RETRY_BACKOFF"
    },
    "dead_letter_queue_max_age": {
        "type": "int",
        "default": 86400,
        "description": "Maximum age in seconds for messages in the dead letter queue.",
        "env_var": "DEAD_LETTER_QUEUE_MAX_AGE"
    },
    "dead_letter_queue_max_messages": {
        "type": "int",
        "default": 1000,
        "description": "Maximum number of messages in the dead letter queue.",
        "env_var": "DEAD_LETTER_QUEUE_MAX_MESSAGES"
    },
    "dead_letter_queue_retention_period": {
        "type": "int",
        "default": 30,
        "description": "Retention period in days for messages in the dead letter queue.",
        "env_var": "DEAD_LETTER_QUEUE_RETENTION_PERIOD"
    },
    "dead_letter_queue_cleanup_interval": {
        "type": "int",
        "default": 3600,
        "description": "Interval in seconds for cleaning up the dead letter queue.",
        "env_var": "DEAD_LETTER_QUEUE_CLEANUP_INTERVAL"
    },
    "dead_letter_queue_cleanup_threshold": {
        "type": "int",
        "default": 500,
        "description": "Threshold for cleaning up the dead letter queue.",
        "env_var": "DEAD_LETTER_QUEUE_CLEANUP_THRESHOLD"
    },
    "dead_letter_queue_log_level": {
        "type": "str",
        "default": "INFO",
        "description": "Log level for dead letter queue operations (DEBUG/INFO/WARN).",
        "enum": ["DEBUG", "INFO", "WARN"],
        "env_var": "DEAD_LETTER_QUEUE_LOG_LEVEL"
    },
    "dead_letter_queue_log_file": {
        "type": "str",
        "default": "dead_letter.log",
        "description": "File path for logging dead letter queue operations.",
        "env_var": "DEAD_LETTER_QUEUE_LOG_FILE"
    },
    "dead_letter_queue_log_max_size": {
        "type": "int",
        "default": 10485760,
        "description": "Maximum size in bytes for the dead letter queue log file.",
        "env_var": "DEAD_LETTER_QUEUE_LOG_MAX_SIZE"
    },
    "dead_letter_queue_log_backup_count": {
        "type": "int",
        "default": 5,
        "description": "Number of backup log files to keep for the dead letter queue.",
        "env_var": "DEAD_LETTER_QUEUE_LOG_BACKUP_COUNT"
    },
    "dead_letter_queue_log_compression": {
        "type": "bool",
        "default": True,
        "description": "Enable compression for dead letter queue log files.",
        "env_var": "DEAD_LETTER_QUEUE_LOG_COMPRESSION"
    },
    "dead_letter_queue_log_rotation": {
        "type": "str",
        "default": "daily",
        "description": "Log rotation schedule for dead letter queue logs (daily/weekly/monthly).
        "enum": ["daily", "weekly", "monthly"],
        "env_var": "DEAD_LETTER_QUEUE_LOG_ROTATION"
    },
    "dead_letter_queue_log_format": {
        "type": "str",
        "default": "%(asctime)s - %(levelname)s - %(message)s",
        "description": "Log format for dead letter queue operations.",
        "env_var": "DEAD_LETTER_QUEUE_LOG_FORMAT"
    },
    "dead_letter_queue_log_datefmt": {
        "type": "str",
        "default": "%Y-%m-%d %H:%M:%S",
        "description": "Date format for dead letter queue logs.",
        "env_var": "DEAD_LETTER_QUEUE_LOG_DATEFMT"
    },
    "dead_letter_queue_log_handler": {
        "type": "str",
        "default": "file",
        "description": "Log handler for dead letter queue logs (file/console).
        "enum": ["file", "console"],
        "env_var": "DEAD_LETTER_QUEUE_LOG_HANDLER"
    },
    "dead_letter_queue_log_level_console": {
        "type": "str",
        "default": "WARN",
        "description": "Log level for console output of dead letter queue logs (DEBUG/INFO/WARN).
        "enum": ["DEBUG", "INFO", "WARN"],
        "env_var": "DEAD_LETTER_QUEUE_LOG_LEVEL_CONSOLE"
    },
    "dead_letter_queue_log_file_encoding": {
        "type": "str",
        "default": "utf-8",
        "description": "Encoding for dead letter queue log files.",
        "env_var": "DEAD_LETTER_QUEUE_LOG_FILE_ENCODING"
    },
    "dead_letter_queue_log_file_mode": {
        "type": "str",
        "default": "a",
        "description": "File mode for dead letter queue log files (a for append, w for write).
        "env_var": "DEAD_LETTER_QUEUE_LOG_FILE_MODE"
    },
    "dead_letter_queue_log_file_permissions": {
        "type": "int",
        "default": 0o644,
        "description": "File permissions for dead letter queue log files.",
        "env_var": "DEAD_LETTER_QUEUE_LOG_FILE_PERMISSIONS"
    },
    "dead_letter_queue_log_file_owner": {
        "type": "str",
        "default": "nobody",
        "description": "Owner of dead letter queue log files.",
        "env_var": "DEAD_LETTER_QUEUE_LOG_FILE_OWNER"
    },
    "dead_letter_queue_log_file_group": {
        "type": "str",
        "default": "nogroup",
        "description": "Group of dead letter queue log files.",
        "env_var": "DEAD_LETTER_QUEUE_LOG_FILE_GROUP"
    },
    "dead_letter_queue_log_file_timezone": {
        "type": "str",
        "default": "UTC",
        "description": "Timezone for dead letter queue logs.",
        "env_var": "DEAD_LETTER_QUEUE_LOG_FILE_TIMEZONE"
    },
    "dead_letter_queue_log_file_timezone_offset": {
        "type": "int",
        "default": 0,
        "description": "Timezone offset in seconds for dead letter queue logs.",
        "env_var": "DEAD_LETTER_QUEUE_LOG_FILE_TIMEZONE_OFFSET"
    },
    "dead_letter_queue_log_file_timezone_abbreviation": {
        "type": "str",
        "default": "UTC",
        "description": "Timezone abbreviation for dead letter queue logs.",
        "env_var": "DEAD_LETTER_QUEUE_LOG_FILE_TIMEZONE_ABBREVIATION"
    },
    "dead_letter_queue_log_file_timezone_name": {
        "type": "str",
        "default": "UTC",
        "description": "Timezone name for dead letter queue logs.",
        "env_var": "DEAD_LETTER_QUEUE_LOG_FILE_TIMEZONE_NAME"
    },
    "dead_letter_queue_log_file_timezone_dst": {
        "type": "bool",
        "default": False,
        "description": "Enable daylight saving time for dead letter queue logs.",
        "env_var": "DEAD_LETTER_QUEUE_LOG_FILE_TIMEZONE_DST"
    }
}

def get_schema() -> Dict[str, Dict]:
    """Return the full configuration schema."""
    return _SCHEMA

def validate_value(key: str, value: Any) -> Any:
    """Validate and coerce a single value against the schema for a given key."""
    schema = _SCHEMA.get(key)
    if not schema:
        raise ConfigValidationError(key, value, f"Key '{key}' not found in schema")

    # Handle enum type
    if schema.get("type") == "enum":
        if not isinstance(value, str):
            raise ConfigValidationError(key, value, f"Value must be a string for enum type '{key}'")
        if value not in schema.get("enum", []):
            raise ConfigValidationError(key, value, f"Invalid enum value for '{key}'")
        return value

    # Handle bool type
    if schema.get("type") == "bool":
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            value = value.lower()
            if value in ["true", "yes", "1", "on"]:
                return True
            elif value in ["false", "no", "0", "off"]:
                return False
            else:
                raise ConfigValidationError(key, value, f"Invalid boolean value for '{key}'")
        else:
            raise ConfigValidationError(key, value, f"Value must be a boolean or string for type '{key}'")

    # Handle int type
    if schema.get("type") == "int":
        if isinstance(value, int):
            return value
        elif isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                raise ConfigValidationError(key, value, f"Invalid integer value for '{key}'")
        else:
            raise ConfigValidationError(key, value, f"Value must be an integer or string for type '{key}'")

    # Handle float type
    if schema.get("type") == "float":
        if isinstance(value, float):
            return value
        elif isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise ConfigValidationError(key, value, f"Invalid float value for '{key}'")
        else:
            raise ConfigValidationError(key, value, f"Value must be a float or string for type '{key}'")

    # Handle string type
    if schema.get("type") == "str":
        if isinstance(value, str):
            return value
        else:
            raise ConfigValidationError(key, value, f"Value must be a string for type '{key}'")

    # Handle default case
    raise ConfigValidationError(key, value, f"Unsupported type '{schema.get('type', 'unknown')}" for '{key}'")

def load_config(config_file: Optional[str] = None, env_vars: Optional[Dict[str, str]] = None, cli_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load configuration from all sources in priority order: CLI > env vars > config file > defaults."""
    config = {}

    # Load CLI arguments first
    if cli_args:
        for key, value in cli_args.items():
            if key in _SCHEMA:
                config[key] = value

    # Load environment variables next
    if env_vars is None:
        env_vars = os.environ
    for key, value in env_vars.items():
        if key in _SCHEMA:
            config[key] = value

    # Load config file next, if provided
    if config_file:
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            for key, value in file_config.items():
                if key in _SCHEMA:
                    config[key] = value
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file '{config_file}' not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in config file '{config_file}'")

    # Load defaults last
    for key, schema in _SCHEMA.items():
        if key not in config:
            config[key] = schema.get("default")

    # Validate all values
    for key, value in config.items():
        if key in _SCHEMA:
            try:
                config[key] = validate_value(key, value)
            except ConfigValidationError as e:
                raise ConfigValidationError(e.key, e.value, f"Validation failed for '{e.key}'") from e

    return config