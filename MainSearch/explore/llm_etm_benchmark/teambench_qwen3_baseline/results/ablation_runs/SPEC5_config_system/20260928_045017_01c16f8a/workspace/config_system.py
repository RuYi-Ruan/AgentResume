from typing import Any, Dict, Optional

# ConfigValidationError class
class ConfigValidationError(ValueError):
    """Raised when a config value fails validation."""
    pass

# Schema definition
_SCHEMA: dict[str, dict] = {
    # Schema entries will be populated based on the full specification provided by the Planner.
    # Example structure:
    # "host": {"type": "string", "default": "0.0.0.0"},
    # "port": {"type": "int", "default": 6155},
    # "log_level": {"type": "enum", "allowed": ["DEBUG", "INFO", "WARN", "ERROR"], "default": "WARN"}
}

# Load configuration function
def load_config(
    config_file: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
    cli_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load and validate configuration from all sources.

    Priority (highest first): cli_args > env_vars > config_file > defaults
    """
    config = {}

    # Apply CLI arguments first
    if cli_args:
        config.update(cli_args)

    # Apply environment variables
    if env_vars:
        config.update(env_vars)

    # Apply config file
    if config_file:
        try:
            with open(config_file, 'r') as f:
                config.update(json.load(f))
        except (IOError, json.JSONDecodeError) as e:
            raise ConfigValidationError(f"Failed to read config file: {e}") from e

    # Apply defaults for missing keys
    for key, spec in _SCHEMA.items():
        if key not in config:
            config[key] = spec.get('default', None)

    # Validate and coerce values
    for key, value in config.items():
        try:
            config[key] = validate_value(key, value)
        except ConfigValidationError as e:
            raise ConfigValidationError(f"Validation failed for key '{key}': {e}") from e

    return config

# Get schema function
def get_schema() -> dict:
    """Return the config schema."""
    return _SCHEMA.copy()

# Validate and coerce value function
def validate_value(key: str, value: Any) -> Any:
    """Validate and coerce a single value for the given config key."""
    spec = _SCHEMA.get(key)
    if not spec:
        raise ConfigValidationError(f"Unknown config key: {key}")

    # Type coercion
    type_coercion = spec.get('type_coercion', 'strict')
    if type_coercion == 'strict':
        if spec['type'] == 'int' and isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                raise ConfigValidationError(f"Value '{value}' cannot be coerced to int for key '{key}'")
        elif spec['type'] == 'float' and isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                raise ConfigValidationError(f"Value '{value}' cannot be coerced to float for key '{key}'")
        elif spec['type'] == 'bool' and isinstance(value, str):
            if value.lower() in ('true', 'yes', '1'):
                value = True
            elif value.lower() in ('false', 'no', '0'):
                value = False
            else:
                raise ConfigValidationError(f"Value '{value}' cannot be coerced to bool for key '{key}'")
        elif spec['type'] == 'enum' and value not in spec.get('allowed', []):
            raise ConfigValidationError(f"Value '{value}' is not allowed for enum key '{key}'")
    elif type_coercion == 'lenient':
        # Lenient coercion logic can be implemented here if needed
        pass

    # Value validation
    if spec['type'] == 'int' and not isinstance(value, int):
        raise ConfigValidationError(f"Value '{value}' is not an integer for key '{key}'")
    elif spec['type'] == 'float' and not isinstance(value, float):
        raise ConfigValidationError(f"Value '{value}' is not a float for key '{key}'")
    elif spec['type'] == 'bool' and not isinstance(value, bool):
        raise ConfigValidationError(f"Value '{value}' is not a boolean for key '{key}'")
    elif spec['type'] == 'enum' and value not in spec.get('allowed', []):
        raise ConfigValidationError(f"Value '{value}' is not allowed for enum key '{key}'")

    return value