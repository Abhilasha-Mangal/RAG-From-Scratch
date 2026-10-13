"""
Utility: Load and validate config.yaml throughout the pipeline.
"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: str | None = None) -> dict:
    """Load config.yaml and resolve env-var secrets."""
    cfg_path = Path(path) if path else _CONFIG_PATH
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_api_key(env_var: str) -> str | None:
    """Safely read an API key from the environment."""
    key = os.getenv(env_var)
    if not key:
        print(f"[warn] Environment variable '{env_var}' is not set.")
    return key
