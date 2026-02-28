import os
import time
from typing import Dict, Any

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


if nn is not None:
    class _ResidualBlock(nn.Module):
        def __init__(self, width: int, dropout: float):
            super().__init__()
            self.norm1 = nn.LayerNorm(width)
            self.lin1 = nn.Linear(width, width)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(width)
            self.lin2 = nn.Linear(width, width)

        def forward(self, x):
            identity = x
            out = self.norm1(x)
            out = self.lin1(out)
            out = self.act(out)
            out = self.drop(out)
            out = self.norm2(out)
            out = self.lin2(out)
            return identity + out


    class _DecisionNet(nn.Module):
        def __init__(self, input_dim: int, width: int, depth: int, action_dim: int, dropout: float):
            super().__init__()
            self.in_proj = nn.Linear(input_dim, width)
            self.blocks = nn.ModuleList([_ResidualBlock(width, dropout) for _ in range(max(1, depth - 2))])
            self.out_norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, action_dim)

        def forward(self, x):
            out = self.in_proj(x)
            for block in self.blocks:
                out = block(out)
            out = self.out_norm(out)
            return self.head(out)
else:
    class _DecisionNet:  # pragma: no cover - fallback when torch is absent
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch not available")


class DecisionPolicy:
    ACTIONS = [
        "delegate",
        "clarify_first",
        "memory_hint",
        "llm_generate",
        "simple_reply",
        "proactive_prompt",
        "autonomy_plan",
        "safety_check",
        "code_intel_query",
        "research_query",
        "tested_apply",
    ]

    def __init__(self):
        self.enabled = os.environ.get("EVOAI_ENABLE_DECISION_LAYER", "1").lower() in ("1", "true", "yes")
        self.allow_autonomy = os.environ.get("EVOAI_DECISION_ALLOW_AUTONOMY", "1").lower() in ("1", "true", "yes")
        self.timeout_ms = int(os.environ.get("EVOAI_DECISION_TIMEOUT_MS", "40"))
        self.depth = int(os.environ.get("EVOAI_DECISION_DEPTH", "12"))
        self.width = int(os.environ.get("EVOAI_DECISION_WIDTH", "512"))
        self.dropout = float(os.environ.get("EVOAI_DECISION_DROPOUT", "0.05"))
        self.input_dim = int(os.environ.get("EVOAI_DECISION_INPUT_DIM", "16"))
        self.model_path = os.environ.get(
            "EVOAI_DECISION_MODEL_PATH",
            os.path.join("data", "decision_policy", "model.pt"),
        )
        self.model_loaded = False

        self.model = None
        if torch is not None and nn is not None and self.enabled:
            try:
                self.model = _DecisionNet(
                    input_dim=self.input_dim,
                    width=self.width,
                    depth=self.depth,
                    action_dim=len(self.ACTIONS),
                    dropout=self.dropout,
                )
                if os.path.isfile(self.model_path):
                    try:
                        state = torch.load(self.model_path, map_location="cpu")
                        if isinstance(state, dict) and "state_dict" in state:
                            self.model.load_state_dict(state["state_dict"], strict=False)
                            self.model_loaded = True
                        elif isinstance(state, dict):
                            self.model.load_state_dict(state, strict=False)
                            self.model_loaded = True
                    except Exception:
                        self.model_loaded = False
                self.model.eval()
            except Exception:
                self.model = None

    def _feature_vector(self, text: str, engine) -> list[float]:
        t = text or ""
        t_stripped = t.strip()
        token_count = len(t_stripped.split())
        char_count = len(t_stripped)
        is_question = 1.0 if "?" in t_stripped else 0.0
        is_repeat = 1.0 if getattr(engine, "last_user", None) == t_stripped else 0.0
        has_plugins = 1.0 if bool(getattr(engine, "plugins", [])) else 0.0
        has_llm = 1.0 if bool(getattr(engine, "llm_model", None) and getattr(engine, "llm_tokenizer", None)) else 0.0
        has_memory = 1.0 if bool(getattr(engine, "memory", [])) else 0.0
        corpus_size = float(len(getattr(engine, "corpus_texts", []) or []))
        similarity_threshold = float(getattr(engine, "similarity_threshold", 0.7))
        has_embeddings = 1.0 if getattr(engine, "embeddings_cache", None) is not None else 0.0
        contains_pronoun = 1.0 if any(w in t_stripped.lower().split() for w in ("it", "this", "that", "they")) else 0.0
        contains_help = 1.0 if "help" in t_stripped.lower() else 0.0
        contains_debug = 1.0 if "debug" in t_stripped.lower() or "error" in t_stripped.lower() else 0.0
        responder_smart = 1.0 if getattr(engine, "responder", None).__class__.__name__.lower().startswith("smart") else 0.0
        memory_size = float(len(getattr(engine, "memory", []) or []))

        raw = [
            float(token_count),
            float(char_count),
            is_question,
            is_repeat,
            has_plugins,
            has_llm,
            has_memory,
            corpus_size,
            similarity_threshold,
            has_embeddings,
            contains_pronoun,
            contains_help,
            contains_debug,
            responder_smart,
            memory_size,
            1.0,
        ]
        return raw[: self.input_dim] + [0.0] * max(0, self.input_dim - len(raw))

    def decide(self, text: str, engine) -> Dict[str, Any]:
        start = time.perf_counter()
        if not self.enabled:
            return {"action": "delegate", "confidence": 1.0, "elapsed_ms": 0.0, "reason": "disabled"}

        features = self._feature_vector(text, engine)

        # Heuristic priors keep behavior stable even without trained weights.
        logits = {action: 0.0 for action in self.ACTIONS}
        if features[3] > 0:  # repeat
            logits["clarify_first"] += 0.15
            logits["delegate"] += 0.20
        if features[2] > 0 and features[10] > 0:  # question + ambiguous pronoun
            logits["clarify_first"] += 0.35
        if features[5] > 0 and features[2] > 0:  # has llm + question
            logits["llm_generate"] += 0.2
        if features[6] > 0 and features[9] > 0:  # has memory + embeddings
            logits["memory_hint"] += 0.2
        if self.allow_autonomy and features[2] > 0:
            logits["proactive_prompt"] += 0.05

        lowered = (text or "").strip().lower()
        if any(k in lowered for k in ("plan", "objective", "goal", "autonomous")):
            logits["autonomy_plan"] += 0.35
        if any(k in lowered for k in ("safety", "guardrail", "risk", "policy")):
            logits["safety_check"] += 0.35
        if any(k in lowered for k in ("code", "symbol", "function", "optimize")):
            logits["code_intel_query"] += 0.30
        if any(k in lowered for k in ("research", "internet", "web", "docs")):
            logits["research_query"] += 0.30
        if any(k in lowered for k in ("apply patch", "rewrite code", "self modify", "optimize code")):
            logits["tested_apply"] += 0.45

        logits["delegate"] += 0.4

        if self.model is not None and torch is not None:
            try:
                with torch.no_grad():
                    x = torch.tensor([features], dtype=torch.float32)
                    out = self.model(x)[0].tolist()
                for idx, action in enumerate(self.ACTIONS):
                    logits[action] += float(out[idx])
            except Exception:
                pass

        ranked = sorted(logits.items(), key=lambda kv: kv[1], reverse=True)
        action, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else best
        confidence = max(0.0, min(1.0, 0.5 + (best - second) / 4.0))

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms > max(1, self.timeout_ms):
            return {
                "action": "delegate",
                "confidence": 0.0,
                "elapsed_ms": elapsed_ms,
                "reason": "timeout",
            }

        return {
            "action": action,
            "confidence": confidence,
            "elapsed_ms": elapsed_ms,
            "reason": "ok",
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "allow_autonomy": self.allow_autonomy,
            "depth": self.depth,
            "width": self.width,
            "dropout": self.dropout,
            "input_dim": self.input_dim,
            "model_path": self.model_path,
            "model_loaded": self.model_loaded,
        }
