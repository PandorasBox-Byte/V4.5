import os
from typing import Any, Dict, Tuple


class SafetyGate:
    """Centralized runtime guard for autonomous and high-risk actions."""

    NETWORK_ACTIONS = {"research_query", "external_backend"}
    SELF_MODIFY_ACTIONS = {"tested_apply"}
    AUTONOMY_ACTIONS = {"autonomy_plan", "tested_apply", "code_intel_query", "research_query"}

    def __init__(self) -> None:
        self.enabled = os.environ.get("EVOAI_SAFETY_GATE", "1").lower() in ("1", "true", "yes")
        self.allow_network_actions = os.environ.get("EVOAI_ALLOW_NETWORK_ACTIONS", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        self.allow_self_modify = os.environ.get("EVOAI_ALLOW_SELF_MODIFY", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self.allow_autonomy_actions = os.environ.get("EVOAI_ALLOW_AUTONOMY_ACTIONS", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        self.max_input_chars = int(os.environ.get("EVOAI_MAX_ACTION_INPUT_CHARS", "12000"))

    def evaluate(self, action: str, text: str, _engine: Any = None) -> Tuple[bool, str]:
        action_name = str(action or "delegate")
        content = str(text or "")

        if not self.enabled:
            return True, "safety_disabled"

        if self.max_input_chars > 0 and len(content) > self.max_input_chars:
            return False, "input_too_large"

        if action_name in self.AUTONOMY_ACTIONS and not self.allow_autonomy_actions:
            return False, "autonomy_disabled"

        if action_name in self.NETWORK_ACTIONS and not self.allow_network_actions:
            return False, "network_disabled"

        if action_name in self.SELF_MODIFY_ACTIONS and not self.allow_self_modify:
            return False, "self_modify_disabled"

        return True, "ok"

    def metadata(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "allow_network_actions": self.allow_network_actions,
            "allow_self_modify": self.allow_self_modify,
            "allow_autonomy_actions": self.allow_autonomy_actions,
            "max_input_chars": self.max_input_chars,
        }
