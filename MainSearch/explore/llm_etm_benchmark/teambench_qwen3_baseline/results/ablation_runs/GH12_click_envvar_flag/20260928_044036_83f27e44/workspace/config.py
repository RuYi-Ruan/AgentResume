"""Configuration loader for deploytool."""
import os


DEFAULT_CONFIG = {
    "deploy_target": "production",
    "timeout": 30,
    "retries": 3,
}


def load_config() -> dict:
    """Load configuration, with optional environment variable overrides."""
    config = dict(DEFAULT_CONFIG)
    env_val = os.environ.get("DEPLOY_TARGET")
    if env_val:
        config["deploy_target"] = env_val
    return config
