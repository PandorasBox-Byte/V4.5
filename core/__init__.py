"""Core package: unified engine architecture.

Exports key components for integrated system access. All core modules are
importable directly (e.g., from core.engine_template import Engine), and collectively
form a unified, interwoven system with the Engine as the central coordinator.

Key modules:
- engine_template: Engine (main coordinator)
- decision_policy: DecisionPolicy (intent routing)
- safety_gate: SafetyGate (autonomy governance)
- code_assistant: CodeAssistant (coding workflows)
- autonomy_tools: CodeIntelToolkit, ResearchToolkit
- tested_apply: TestedApplyOrchestrator
- memory: Conversation persistence
- embeddings_cache: Vector cache
- launcher: Startup entry point
- tui: Text UI
- api_server: REST API
- trainer: Model training functions
"""

# ============================================================================
# Compatibility shims
# ============================================================================

try:
	import huggingface_hub as _hfh
	if not hasattr(_hfh, "cached_download"):
		# prefer top-level hf_hub_download, otherwise try utils
		target = getattr(_hfh, "hf_hub_download", None)
		if target is None:
			try:
				from huggingface_hub.utils import hf_hub_download as target
			except Exception:
				target = None

		if target is not None:
			def cached_download(*args, **kwargs):
				# map simple signature to hf_hub_download where possible
				return target(*args, **kwargs)

			setattr(_hfh, "cached_download", cached_download)
except Exception:
	# don't prevent imports if huggingface_hub not installed
	pass


# ============================================================================
# Convenience imports
# ============================================================================

# Core engine
from core.engine_template import Engine, SimpleResponder, SmartResponder
from core.decision_policy import DecisionPolicy
from core.safety_gate import SafetyGate
from core.code_assistant import CodeAssistant

__all__ = [
	"Engine",
	"SimpleResponder",
	"SmartResponder",
	"DecisionPolicy",
	"SafetyGate",
	"CodeAssistant",
]
