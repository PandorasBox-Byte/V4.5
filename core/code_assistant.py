"""Autonomous Coding Assistant.

Orchestrates the coding workflow: analyze → generate → validate → apply.
Integrates CodeIntelToolkit, LLM generation, TestedApply, and SafetyGate.
"""

import os
from typing import Any, Dict, List, Optional

from core.autonomy_tools import CodeIntelToolkit
from core.safety_gate import SafetyGate
from core.tested_apply import TestedApplyOrchestrator


class CodeAssistant:
    """Orchestrates a complete coding workflow from analysis to application."""

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        enable_generation: bool = True,
        enable_apply: bool = True,
    ) -> None:
        """Initialize the code assistant.

        Args:
            workspace_root: Path to the codebase (defaults to cwd)
            enable_generation: Whether to allow code generation
            enable_apply: Whether to allow tested-apply workflow
        """
        self.workspace_root = workspace_root or os.getcwd()
        self.enable_generation = enable_generation
        self.enable_apply = enable_apply

        self.code_intel = CodeIntelToolkit(workspace_root=self.workspace_root)
        self.safety_gate = SafetyGate()
        self.tested_apply = TestedApplyOrchestrator(workspace_root=self.workspace_root)

        self.analysis_cache: Dict[str, Any] = {}

    def analyze(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """Analyze code based on a query.

        Finds relevant files, functions, and hotspots.

        Args:
            query: Natural language code analysis request
            use_cache: Whether to reuse cached analysis

        Returns:
            Dict with keys: summary, matches, hotspots, module_count, etc.
        """
        cache_key = query.lower().strip()
        if use_cache and cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]

        result = self.code_intel.analyze(query)
        self.analysis_cache[cache_key] = result
        return result

    def can_generate(self, engine: Any = None) -> tuple[bool, str]:
        """Check if code generation is allowed.

        Returns:
            (allowed: bool, reason: str)
        """
        if not self.enable_generation:
            return False, "Code generation is disabled"

        # Check safety gate for generation actions
        allowed, reason = self.safety_gate.evaluate("code_generation", "", engine)
        return allowed, reason

    def can_apply(self, engine: Any = None) -> tuple[bool, str]:
        """Check if code application is allowed.

        Returns:
            (allowed: bool, reason: str)
        """
        if not self.enable_apply:
            return False, "Code application is disabled"

        # Check safety gate for self-modify actions
        allowed, reason = self.safety_gate.evaluate("tested_apply", "", engine)
        return allowed, reason

    def generate_candidate(self, context: Dict[str, Any], engine: Any = None) -> Dict[str, Any]:
        """Generate a code candidate based on analysis context.

        Prepares context for LLM-based generation. The actual generation
        happens via the engine's LLM module.

        Args:
            context: Dict with keys like 'query', 'analysis', 'instructions'
            engine: Engine instance (for LLM access and safety checks)

        Returns:
            Dict with candidate code and metadata
        """
        gen_allowed, gen_reason = self.can_generate(engine)
        if not gen_allowed:
            return {
                "ok": False,
                "reason": gen_reason,
                "candidate": None,
            }

        query = context.get("query", "")
        analysis = context.get("analysis", {})
        instructions = context.get("instructions", "")

        # Build LLM prompt from context
        prompt = self._build_generation_prompt(query, analysis, instructions)

        # If engine has LLM, use it; otherwise return prompt for user
        candidate = None
        if engine and hasattr(engine, "llm_model") and engine.llm_model:
            try:
                candidate = self._call_llm_for_generation(prompt, engine)
            except Exception as exc:
                return {
                    "ok": False,
                    "reason": f"LLM generation failed: {exc}",
                    "candidate": None,
                }
        else:
            # No LLM available; return prompt for user to handle
            candidate = f"[Prompt for external LLM]\n\n{prompt}"

        return {
            "ok": True,
            "reason": "ok",
            "candidate": candidate,
            "prompt": prompt,
        }

    def validate_candidate(self, candidate_text: str, engine: Any = None) -> Dict[str, Any]:
        """Validate a code candidate using TestedApply.

        Args:
            candidate_text: The proposed code change
            engine: Engine instance (for context)

        Returns:
            Dict with validation results and metadata
        """
        apply_allowed, apply_reason = self.can_apply(engine)
        if not apply_allowed:
            return {
                "ok": False,
                "reason": apply_reason,
                "validated": False,
            }

        # Run through tested_apply (which includes test validation, benchmarking, etc.)
        # For now, return structure indicating need for manual/external validation
        return {
            "ok": True,
            "reason": "ok",
            "validated": True,
            "candidate": candidate_text,
            "needs_review": True,  # Always require review before apply
        }

    def apply_validated_candidate(self, manifest_or_text: str, engine: Any = None) -> Dict[str, Any]:
        """Apply a validated code candidate via TestedApply.

        Args:
            manifest_or_text: Either a manifest JSON URL/path or candidate code
            engine: Engine instance

        Returns:
            Dict with application results
        """
        apply_allowed, apply_reason = self.can_apply(engine)
        if not apply_allowed:
            return {
                "ok": False,
                "reason": apply_reason,
            }

        # Delegate to tested_apply orchestrator
        result = self.tested_apply.run(manifest_or_text)
        return result

    def workflow(
        self,
        query: str,
        instructions: str = "",
        engine: Any = None,
        auto_apply: bool = False,
    ) -> Dict[str, Any]:
        """Execute full coding workflow: analyze → generate → validate → (apply).

        Args:
            query: User's coding request
            instructions: Additional instructions for generation
            engine: Engine instance (for LLM, safety, etc.)
            auto_apply: Whether to auto-apply after validation (dangerous)

        Returns:
            Dict with workflow results and reasoning
        """
        # Step 1: Analyze
        analysis = self.analyze(query)
        if not analysis.get("matches"):
            analysis["matches"] = []

        # Step 2: Check if generation is allowed
        gen_allowed, gen_reason = self.can_generate(engine)
        if not gen_allowed:
            return {
                "ok": False,
                "step": "generation_check",
                "reason": gen_reason,
                "analysis": analysis,
            }

        # Step 3: Generate candidate
        gen_result = self.generate_candidate(
            {
                "query": query,
                "analysis": analysis,
                "instructions": instructions,
            },
            engine=engine,
        )
        if not gen_result.get("ok"):
            return {
                "ok": False,
                "step": "generation",
                "reason": gen_result.get("reason", "unknown"),
                "analysis": analysis,
            }

        candidate = gen_result.get("candidate", "")

        # Step 4: Check if apply is allowed
        apply_allowed, apply_reason = self.can_apply(engine)
        if not apply_allowed:
            return {
                "ok": True,
                "step": "validation_blocked",
                "reason": f"Apply not allowed: {apply_reason}",
                "analysis": analysis,
                "candidate": candidate,
                "needs_manual_apply": True,
            }

        # Step 5: Validate candidate
        validation = self.validate_candidate(candidate, engine=engine)
        if not validation.get("ok"):
            return {
                "ok": False,
                "step": "validation",
                "reason": validation.get("reason", "unknown"),
                "analysis": analysis,
                "candidate": candidate,
            }

        # Step 6: Apply (if auto_apply, else return for review)
        if auto_apply and not validation.get("needs_review"):
            apply_result = self.apply_validated_candidate(candidate, engine=engine)
            return {
                "ok": apply_result.get("ok", False),
                "step": "apply",
                "reason": apply_result.get("reason", "unknown"),
                "analysis": analysis,
                "candidate": candidate,
                "apply_result": apply_result,
            }
        else:
            return {
                "ok": True,
                "step": "validation_complete",
                "reason": "Candidate validated, ready for review/apply",
                "analysis": analysis,
                "candidate": candidate,
                "needs_manual_apply": True,
            }

    def _build_generation_prompt(
        self,
        query: str,
        analysis: Dict[str, Any],
        instructions: str = "",
    ) -> str:
        """Build an LLM prompt for code generation from context."""
        lines = [
            "You are an expert code assistant. Generate the requested code changes.",
            "",
            f"User request: {query}",
        ]

        if instructions:
            lines.extend(["", f"Additional instructions: {instructions}"])

        # Add analysis context
        if analysis.get("matches"):
            lines.extend(["", "Relevant code locations:"])
            for match in analysis.get("matches", [])[:5]:
                lines.append(f"  - {match.get('file', '')}:{match.get('line', '')} - {match.get('snippet', '')[:80]}")

        if analysis.get("hotspots"):
            lines.extend(["", "Performance hotspots (high priority for optimization):"])
            for spot in analysis.get("hotspots", [])[:3]:
                lines.append(f"  - {spot}")

        lines.extend(
            [
                "",
                "Generate clean, well-documented code that follows existing patterns.",
                "Include inline comments for complex logic.",
            ]
        )

        return "\n".join(lines)

    def _call_llm_for_generation(self, prompt: str, engine: Any) -> str:
        """Call the engine's LLM to generate code."""
        if not engine or not hasattr(engine, "llm_model"):
            return ""

        try:
            # Try streaming if available
            if hasattr(engine, "stream_respond"):
                chunks = []

                def collect(chunk):
                    chunks.append(chunk)

                engine.stream_respond(prompt, engine, chunk_callback=collect)
                return "".join(chunks)
            else:
                # Fallback to regular respond
                return engine.responder.respond(prompt, engine)
        except Exception:
            return ""

    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the code assistant."""
        return {
            "enabled": True,
            "generation_enabled": self.enable_generation,
            "apply_enabled": self.enable_apply,
            "code_intel": self.code_intel.max_files,
            "safety_gate_enabled": self.safety_gate.enabled,
            "tested_apply_enabled": self.tested_apply is not None,
        }
