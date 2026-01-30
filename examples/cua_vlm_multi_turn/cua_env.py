from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from typing import Any

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    orjson = None

from slime.utils.types import Sample

import requests
import os

logger = logging.getLogger(__name__)

# Matches the JSON payload emitted between <tool_call> ... </tool_call> tags.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Accept either name; verl uses `calc_geo3k_reward` while the instruction refers to `calc_score`.
SUPPORTED_TOOL_NAMES = {"calc_score", "calc_geo3k_reward"}

import random

class CUAEnv:
    """
    Minimal interaction environment for multi-turn CUA with a scoring tool.

    The model is expected to emit a <tool_call>{...}</tool_call> payload that includes
    an `action` argument. We execute the action and return the result as the next observation.
    """

    def __init__(self, os_env_ip: str, os_env_available_ports: list[int]):
        self.os_env_ip = os_env_ip
        self.os_env_available_ports = os_env_available_ports
        self.os_env_port = random.choice(os_env_available_ports)
        # self.os_env_port = os_available_ports[0] if os_available_ports else 20000

    def reset(self, task_config: dict) -> dict:
        """Reset the OS environment with the given task configuration."""
        self.task_config = task_config
        url = f"http://{self.os_env_ip}:{self.os_env_port}/reset"

        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={"task_config": self.task_config, "timeout": 1000},
                timeout=600
            )
            response.raise_for_status()
            result = response.json()
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to reset environment: {e}")
            raise

    def close(self):
        """No resources to release."""
        return

    # Called during rollout after receiving a model response
    def step(self, action_code: str, vm_id: str):
        url = f"http://{self.os_env_ip}:{self.os_env_port}/step"

        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={"action": action_code, "vm_id": vm_id},
                timeout=600
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to execute step: {e}")
            raise


def build_env(sample: Sample | None = None, args: Any | None = None, **_: Any) -> CUAEnv:
    os_env_ip = os.getenv("OS_ENV_IP", "127.0.0.1")
    os_env_available_ports = [int(port) for port in os.getenv("OS_ENV_AVAILABLE_PORTS", "20000").split(",")]
    return CUAEnv(os_env_ip=os_env_ip, os_env_available_ports=os_env_available_ports)
