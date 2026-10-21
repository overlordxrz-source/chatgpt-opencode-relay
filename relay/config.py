"""Configuration handling for the relay."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path.home() / ".chatgpt-opencode-relay"
DEFAULT_CONFIG = DEFAULT_DIR / "config.json"

DEFAULTS: dict[str, Any] = {
    "chatgpt": {
        "conversation": "",
        "poll_seconds": 3.0,
    },
    "opencode": {
        "bridge_url": "http://127.0.0.1:8787/mcp",
        "bridge_env": str(Path.home() / "tools/opencode-chatgpt-bridge-v2/.env"),
        "default_session": "",
    },
    "limits": {
        "strict_protocol": True,
        "max_exchanges_per_hour": 12,
        "max_exchanges_per_day": 40,
        "max_reply_chars": 6000,
        "agent_timeout_ms": 180000,
        "max_loop_repeats": 3,
    },
    "paths": {
        "state_db": str(DEFAULT_DIR / "state.db"),
        "status_file": str(DEFAULT_DIR / "status.json"),
        "log": str(DEFAULT_DIR / "relay.log"),
    },
    "post_init_prompt": True,
}


def config_path() -> Path:
    return Path(os.environ.get("RELAY_CONFIG", DEFAULT_CONFIG)).expanduser()


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or config_path()
    data = json.loads(json.dumps(DEFAULTS))  # deep copy
    if path.exists():
        stored = json.loads(path.read_text())
        for key, value in stored.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key].update(value)
            else:
                data[key] = value
    return data


def save_config(data: dict[str, Any], path: Path | None = None) -> Path:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    path.chmod(0o600)
    return path


def bridge_token(config: dict[str, Any]) -> str:
    """Read the bridge bearer token from the bridge's own .env file.

    The relay never stores the token itself.
    """
    env_path = Path(config["opencode"]["bridge_env"]).expanduser()
    if not env_path.exists():
        raise RuntimeError(f"bridge env file not found: {env_path}")
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("OPENCODE_BRIDGE_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError(f"OPENCODE_BRIDGE_TOKEN missing in {env_path}")


def path_of(config: dict[str, Any], key: str) -> Path:
    return Path(config["paths"][key]).expanduser()
