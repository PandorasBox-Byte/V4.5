from __future__ import annotations

import os
import atexit
import signal
import sys
import math
import json
import time
from collections import deque
from typing import List

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

    class _CosResult:
        def __init__(self, scores):
            self.scores = scores

        def max(self, dim=1):
            if not self.scores:
                return _Num(0.0), _Num(0)
            best_idx = max(range(len(self.scores)), key=lambda i: self.scores[i])
            return _Num(float(self.scores[best_idx])), _Num(int(best_idx))

    class _Num:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class _UtilFallback:
        @staticmethod
        def cos_sim(query_embedding, corpus_embeddings):
            q = list(query_embedding) if isinstance(query_embedding, (list, tuple)) else []
            rows = corpus_embeddings if isinstance(corpus_embeddings, list) else []
            qnorm = math.sqrt(sum(v * v for v in q)) or 1.0
            scores = []
            for row in rows:
                r = list(row) if isinstance(row, (list, tuple)) else []
                rnorm = math.sqrt(sum(v * v for v in r)) or 1.0
                dot = sum((a * b) for a, b in zip(q, r))
                scores.append(dot / (qnorm * rnorm))
            return _CosResult(scores)

    util = _UtilFallback()
from core.memory import load_memory, save_memory, prune_memory
from core.embeddings_cache import load_embeddings, save_embeddings, clear_cache
from core.github_backend import GitHubBackend
from core.decision_policy import DecisionPolicy
from core.safety_gate import SafetyGate
from core.autonomy_tools import CodeIntelToolkit, ResearchToolkit
from core.tested_apply import TestedApplyOrchestrator
from core.code_assistant import CodeAssistant
from core.brain_monitor import BrainMonitor, install_trace_hook


def _path_candidates(raw: str | None) -> List[str]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(":")]
    return [p for p in parts if p]


def _resolve_local_or_name(candidates: List[str]) -> str | None:
    for name in candidates:
        if os.path.isdir(name):
            return os.path.abspath(name)
    return candidates[0] if candidates else None


class Responder:
    """Base class for response strategies.

    Subclasses should implement ``respond(query, engine)``.  The engine is
    passed so that responders can access memory, embeddings, plugins, etc.
    """

    def respond(self, query: str, engine: "Engine") -> str:
        raise NotImplementedError


class SimpleResponder(Responder):
    """Original ``Engine.respond`` logic extracted into a responder.

    This responder performs the familiar similarity check, saves new memory
    entries, and echoes the input when nothing special is found.  It does not
    consult any plugins.
    """

    def respond(self, text: str, engine: "Engine") -> str:  # pylint: disable=too-many-branches
        if not text or not text.strip():
            return "Please enter something."

        # Compute query embedding
        try:
            query_embedding = engine._encode(text)
        except Exception:  # noqa: BLE001
            query_embedding = None

        # If we have cached embeddings, check similarity first
        if (
            query_embedding is not None
            and engine.embeddings_cache is not None
            and len(engine.corpus_texts) > 0
        ):
            try:
                hits = util.cos_sim(query_embedding, engine.embeddings_cache)
                score, idx = hits.max(dim=1)
                if score.item() > engine.similarity_threshold:
                    chosen = engine.corpus_texts[int(idx.item())].strip()
                    return f"You previously said something similar: '{chosen}'"
            except Exception:  # noqa: BLE001
                pass

        # No high-similarity hit: build reply, record interaction, and update cache
        reply = f"I understand you said: '{text}'"
        engine.record_interaction(text, reply)

        # Update corpus_texts and the on-disk embeddings cache (handled by record_interaction already)
        return reply


class SmartResponder(SimpleResponder):
    """More intelligent strategy that leverages memory and plugins.

    * Checks registered research plugins before falling back to the simple
      reply logic.
    * If a memory entry is highly similar the query, returns a more natural
      acknowledgement and invites elaboration.
    * When an LLM is configured, produces a generative reply based on the
      conversation context.
    """

    def build_prompt(self, text: str, engine: "Engine") -> str:
        # create a simple conversational prompt including a few recent memory
        # entries to give the LLM some context.  This may be extended later.
        prompt = """You are EvoAI, a helpful assistant.\n"""
        recent = engine.memory[-5:]
        for entry in recent:
            prompt += f"User: {entry.get('user','')}\n"
        prompt += f"User: {text}\nEvoAI:"
        return prompt

    def generate_with_model(self, prompt: str, engine: "Engine") -> str | None:
        if not getattr(engine, "llm_model", None) or not getattr(engine, "llm_tokenizer", None):
            return None
        try:
            inputs = engine.llm_tokenizer(prompt, return_tensors="pt")
            # move tensors to the same device as the model
            try:
                model_device = next(engine.llm_model.parameters()).device
            except Exception:
                model_device = getattr(engine, "llm_device", "cpu")

            for k, v in inputs.items():
                try:
                    inputs[k] = v.to(model_device)
                except Exception:
                    pass

            # Prepare generation kwargs with sensible defaults
            gen_kwargs = {
                "max_new_tokens": int(engine.llm_params.get("max_new_tokens", 50)),
                "temperature": float(engine.llm_params.get("temperature", 1.0)),
                "top_k": int(engine.llm_params.get("top_k", 0)),
                "top_p": float(engine.llm_params.get("top_p", 1.0)),
                "do_sample": float(engine.llm_params.get("temperature", 1.0)) > 0,
            }

            # Ensure a pad token id for some models
            pad_id = getattr(engine.llm_tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(engine.llm_tokenizer, "eos_token_id", None)
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = pad_id

            # Merge any additional gen params supplied via llm_params
            extra = engine.llm_params.get("extra_gen_kwargs", {})
            if isinstance(extra, dict):
                gen_kwargs.update(extra)

            outputs = engine.llm_model.generate(**inputs, **gen_kwargs)
            text = engine.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove prompt prefix if present so we return only the generated completion
            if text.startswith(prompt):
                text = text[len(prompt) :]

            return text.strip()
        except Exception:  # noqa: BLE001
            return None

    def stream_respond(self, text: str, engine: "Engine", chunk_callback) -> None:
        """Stream a generated response by invoking the engine's generation
        in a background thread and calling `chunk_callback(chunk, final=False)`
        for each chunk and `chunk_callback(chunk, final=True)` when done.
        """
        # Short-circuit repetition and plugin handling as in `respond`.
        if engine.last_user == text:
            # try to produce a different reply using the LLM if present
            if engine.llm_model and engine.llm_tokenizer:
                prompt = self.build_prompt(text, engine) + "\n(Even the same query should get a fresh answer.)"
                # Run generation in background and stream
                engine.generate_stream(prompt, chunk_callback)
                return
            reply = "You've already said that. Could you expand or change the topic?"
            engine.record_interaction(text, reply)
            chunk_callback(reply, True)
            return

        for plugin in getattr(engine, "plugins", []):
            try:
                if plugin.can_handle(text):
                    result = plugin.handle(text, engine)
                    if result is not None:
                        engine.record_interaction(text, str(result))
                        chunk_callback(result, True)
                        return
            except Exception:  # pragma: no cover - plugin bugs shouldn't kill us
                continue

        if engine.embeddings_cache is not None and engine.corpus_texts:
            try:
                query_emb = engine._encode(text)
                hits = util.cos_sim(query_emb, engine.embeddings_cache)
                score, idx = hits.max(dim=1)
                if score.item() > engine.similarity_threshold:
                    previous = engine.corpus_texts[int(idx.item())].strip()
                    reply = f"I recall you mentioned '{previous}' earlier. Could you tell me more?"
                    engine.record_interaction(text, reply)
                    chunk_callback(reply, True)
                    return
            except Exception:
                pass

        # If LLM is configured, stream generation
        prompt = self.build_prompt(text, engine)
        if engine.llm_model and engine.llm_tokenizer:
            # record interaction when final chunk arrives inside engine.generate_stream
            engine.generate_stream(prompt, chunk_callback, record_user_text=text)
            return

        # fallback to synchronous responder
        reply = super().respond(text, engine)
        chunk_callback(reply, True)

    def respond(self, text: str, engine: "Engine") -> str:
        # first, check if the user is repeating the exact same input as last
        if engine.last_user == text:
            # try to produce a different reply, prefer using the LLM if present
            if engine.llm_model and engine.llm_tokenizer:
                prompt = self.build_prompt(text, engine) + "\n(Even the same query should get a fresh answer.)"
                alt = self.generate_with_model(prompt, engine)
                if alt and alt.strip() != engine.last_reply:
                    engine.record_interaction(text, alt)
                    return alt
            # fallback canned response
            reply = "You've already said that. Could you expand or change the topic?"
            engine.record_interaction(text, reply)
            return reply

        # allow any plugin to short-circuit the response
        for plugin in getattr(engine, "plugins", []):
            try:
                if plugin.can_handle(text):
                    result = plugin.handle(text, engine)
                    if result is not None:
                        engine.record_interaction(text, str(result))
                        return result
            except Exception:  # pragma: no cover - plugin bugs shouldn't kill us
                continue

        # memory-based enhancement: look for similar past entry
        if engine.embeddings_cache is not None and engine.corpus_texts:
            try:
                query_emb = engine._encode(text)
                hits = util.cos_sim(query_emb, engine.embeddings_cache)
                score, idx = hits.max(dim=1)
                if score.item() > engine.similarity_threshold:
                    previous = engine.corpus_texts[int(idx.item())].strip()
                    reply = (
                        f"I recall you mentioned '{previous}' earlier. "
                        "Could you tell me more?"
                    )
                    engine.record_interaction(text, reply)
                    return reply
            except Exception:  # noqa: BLE001
                pass

        # if an LLM is configured, try generating a response
        prompt = self.build_prompt(text, engine)
        gen = self.generate_with_model(prompt, engine)
        if gen:
            engine.record_interaction(text, gen)
            return gen

        # no plugin handled it and no interesting memory hit; fall back
        reply = super().respond(text, engine)
        # ``super`` already recorded the interaction in simple case
        return reply


class Engine:
    """Local AI assistant engine with integrated decision routing and autonomy governance.
    
    Architecture Overview:
    - Decision Policy: Routes user queries to appropriate action handlers based on intent
    - Safety Gate: Validates all autonomous actions against runtime policy
    - CodeAssistant: Specialized orchestrator for coding workflows (analyze→generate→validate→apply)
    - Toolkits: CodeIntelToolkit (code analysis), ResearchToolkit (external research)
    - Orchestrators: TestedApplyOrchestrator (validated code application), TrainerOrchestrator (model training)
    - Persistence: Memory (conversation), EmbeddingsCache (semantic vectors)
    - External: GitHubBackend (external LLM), PluginManager (extensibility)
    - Governance: Safety gates, autonomy budget, audit events, outcome logging
    
    Integration Flow:
    1. User query → respond()
    2. respond() → Decision Policy determines action type
    3. Decision Policy checks Safety Gate for permission
    4. Action routes to handler (code_assist, llm_generate, research_query, etc.)
    5. Handler executes through specialized toolkit/orchestrator
    6. All autonomous actions logged in audit stream
    7. Memory/embeddings updated and persisted
    
    Environment Controls:
    - EVOAI_DECISION_LAYER: Enable/disable decision routing
    - EVOAI_AUTONOMY_PAUSED: Pause all autonomous actions
    - EVOAI_RESPONDER: Choose responder (simple vs smart)
    - EVOAI_BACKEND_PROVIDER: External backend (github)
    """
    def __init__(self, progress_cb=None):
        """Engine constructor.

        progress_cb: optional callable(progress: float, message: str)
        used by the TUI to visualize loading progress.
        """
        # housekeeping state
        self._status = {}
        self.clarifications_asked = set()

        self._progress_cb = progress_cb
        # if thesaurus is enabled, make sure WordNet corpora exist now to avoid
        # delays later during response generation
        if os.environ.get("EVOAI_USE_THESAURUS", "1").lower() in ("1", "true", "yes"):
            try:
                from core.language_utils import ensure_wordnet

                ensure_wordnet()
            except Exception:
                pass
        def _report(frac, msg=""):
            try:
                if callable(self._progress_cb):
                    self._progress_cb(float(frac), str(msg))
            except Exception:
                pass

        self._set_status("startup", "begin")
        _report(0.01, "starting")

        # Avoid writing startup logs directly to stdout when running under the
        # TUI loader, because it can corrupt the curses screen.
        self._startup_quiet = callable(progress_cb) and os.environ.get(
            "EVOAI_STARTUP_STDOUT_LOGS", "0"
        ).lower() not in ("1", "true", "yes")

        def _startup_log(msg: str) -> None:
            if self._startup_quiet:
                return
            try:
                print(msg)
            except Exception:
                pass

        _startup_log("Loading model (Intel CPU optimized)...")
        if torch is not None:
            try:
                torch.set_num_threads(int(os.environ.get("EVOAI_TORCH_THREADS", "8")))
            except Exception:
                pass

        auto_discovery = os.environ.get("EVOAI_AUTO_MODEL_DISCOVERY", "1").lower() in (
            "1",
            "true",
            "yes",
        )

        # allow previously fine-tuned model to override the base name, with
        # optional auto-discovery fallback through common local model dirs.
        emb_candidates: List[str] = []
        emb_primary = os.environ.get("EVOAI_FINETUNED_MODEL", "").strip()
        if emb_primary:
            emb_candidates.append(emb_primary)
        emb_candidates.extend(_path_candidates(os.environ.get("EVOAI_FINETUNED_MODEL_CANDIDATES")))
        if auto_discovery:
            emb_candidates.extend(
                [
                    os.path.join("data", "finetuned-model"),
                    os.path.join("data", "llm_finetuned"),
                    os.path.join("data", "llm_finetuned_debug"),
                ]
            )
        emb_candidates.append(os.environ.get("EVOAI_MODEL", "all-MiniLM-L6-v2"))
        model_name = _resolve_local_or_name(emb_candidates) or "all-MiniLM-L6-v2"
        self._set_status("embeddings_selected", str(model_name))
        # Force CPU device to maintain determinism on Intel Mac
        device = os.environ.get("EVOAI_DEVICE", "cpu")

        self._set_status("config", "loaded")

        self.decision_policy = DecisionPolicy()
        self.safety_gate = SafetyGate()
        self.code_intel_toolkit = CodeIntelToolkit(workspace_root=os.getcwd())
        self.research_toolkit = ResearchToolkit()
        self.tested_apply_orchestrator = TestedApplyOrchestrator(workspace_root=os.getcwd())
        self.code_assistant = CodeAssistant(workspace_root=os.getcwd())
        
        # Initialize brain monitor (tracks file execution in real-time)
        self.brain_monitor = None
        if os.environ.get("EVOAI_ENABLE_BRAIN_MONITOR", "1").lower() in ("1", "true", "yes"):
            try:
                self.brain_monitor = BrainMonitor(workspace_root=os.getcwd())
                install_trace_hook(self.brain_monitor)
            except Exception:
                pass
        
        self.autonomy_paused = os.environ.get("EVOAI_AUTONOMY_PAUSED", "0").lower() in ("1", "true", "yes")
        self.autonomy_budget_max = max(0, int(os.environ.get("EVOAI_AUTONOMY_BUDGET_MAX", "25")))
        self.autonomy_budget_remaining = max(
            0,
            int(os.environ.get("EVOAI_AUTONOMY_BUDGET_REMAINING", str(self.autonomy_budget_max))),
        )
        self._apply_governance_policy_defaults()
        self.audit_max_events = max(10, int(os.environ.get("EVOAI_AUDIT_MAX_EVENTS", "200")))
        self._audit_events = deque(maxlen=self.audit_max_events)
        self.outcome_log_enabled = os.environ.get("EVOAI_ENABLE_OUTCOME_LOG", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self.outcomes_path = os.environ.get("EVOAI_OUTCOMES_PATH", os.path.join("data", "autonomy_outcomes.jsonl"))
        self.decision_max_proactive_per_turn = int(
            os.environ.get("EVOAI_DECISION_MAX_PROACTIVE_PER_TURN", "2")
        )
        self.decision_autonomy_cooldown = int(
            os.environ.get("EVOAI_DECISION_AUTONOMY_COOLDOWN", "2")
        )
        self._turns_since_proactive = self.decision_autonomy_cooldown
        self._set_status("decision_layer", "enabled" if self.decision_policy.enabled else "disabled")
        self._set_status("decision_depth", str(self.decision_policy.depth))
        self._set_status("decision_width", str(self.decision_policy.width))
        self._set_status("decision_model_loaded", str(self.decision_policy.model_loaded).lower())
        self._set_status("decision_model_path", str(self.decision_policy.model_path))
        self._set_status("safety_gate", "enabled" if self.safety_gate.enabled else "disabled")
        self._set_status("safety_allow_network", str(self.safety_gate.allow_network_actions).lower())
        self._set_status("safety_allow_self_modify", str(self.safety_gate.allow_self_modify).lower())
        self._set_status("research_web_enabled", str(self.research_toolkit.allow_web).lower())
        self._set_status("tested_apply_post_pytest", str(self.tested_apply_orchestrator.post_pytest).lower())
        self._set_status("autonomy_paused", str(self.autonomy_paused).lower())
        self._set_status("autonomy_budget_max", str(self.autonomy_budget_max))
        self._set_status("autonomy_budget_remaining", str(self.autonomy_budget_remaining))
        self._set_status("governance_policy_loaded", str(getattr(self, "governance_policy_loaded", False)).lower())
        self._set_status(
            "governance_policy_recommendation",
            str(getattr(self, "governance_policy_recommendation", "none")),
        )
        self._set_status(
            "governance_policy_applied",
            str(getattr(self, "governance_policy_applied", "none")),
        )
        self._set_status("audit_events", "0")
        self._set_status("outcome_log_enabled", str(self.outcome_log_enabled).lower())
        self._set_status("outcomes_path", str(self.outcomes_path))

        _report(0.10, "loading embeddings model")
        if SentenceTransformer is not None:
            self.model = None
            seen = set()
            for candidate in emb_candidates:
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                resolved = os.path.abspath(candidate) if os.path.isdir(candidate) else candidate
                try:
                    self.model = SentenceTransformer(resolved, device=device)
                    model_name = resolved
                    break
                except Exception:
                    continue

            if self.model is not None:
                self._encode = lambda texts, **k: self.model.encode(
                    texts, convert_to_tensor=True, **k
                )
            else:
                def _fallback_encode(texts, **k):
                    if isinstance(texts, str):
                        text = texts
                    elif isinstance(texts, list):
                        return [_fallback_encode(t) for t in texts]
                    else:
                        text = str(texts)
                    vec = [0.0] * 16
                    for idx, ch in enumerate(text.lower()):
                        vec[idx % 16] += (ord(ch) % 31) / 31.0
                    return vec

                self._encode = _fallback_encode
        else:
            self.model = None

            def _fallback_encode(texts, **k):
                if isinstance(texts, str):
                    text = texts
                elif isinstance(texts, list):
                    return [_fallback_encode(t) for t in texts]
                else:
                    text = str(texts)
                vec = [0.0] * 16
                for idx, ch in enumerate(text.lower()):
                    vec[idx % 16] += (ord(ch) % 31) / 31.0
                return vec

            self._encode = _fallback_encode
        _report(0.45, "embeddings model loaded")
        self._set_status("embeddings_model", "ok")

        # Configuration
        self.similarity_threshold = float(
            os.environ.get("EVOAI_SIMILARITY_THRESHOLD", "0.7")
        )
        self.max_memory_entries = int(os.environ.get("EVOAI_MAX_MEMORY", "500"))

        # tracking most recent interaction to avoid repetition
        self.last_user = None
        self.last_reply = None

        # LLM generation parameters (passed through to ``model.generate``)
        self.llm_params = {
            "max_new_tokens": int(os.environ.get("EVOAI_LLM_MAX_TOKENS", "50")),
            # additional spacing, temperature, top_k, top_p, etc. can be
            # provided via env vars if needed
            "temperature": float(os.environ.get("EVOAI_LLM_TEMPERATURE", "1.0")),
            "top_k": int(os.environ.get("EVOAI_LLM_TOP_K", "0")),
            "top_p": float(os.environ.get("EVOAI_LLM_TOP_P", "1.0")),
        }

        # Persistent memory and in-memory embedding cache
        self.memory = load_memory()
        self.corpus_texts = [entry.get("user", "") for entry in self.memory]
        self._persist_every = max(1, int(os.environ.get("EVOAI_PERSIST_EVERY", "2")))
        self._persist_max_delay_sec = max(
            0.0,
            float(os.environ.get("EVOAI_PERSIST_MAX_DELAY_SEC", "2.0")),
        )
        self._pending_persist_turns = 0
        self._dirty_memory = False
        self._dirty_embeddings = False
        self._last_persist_ts = time.monotonic()
        self._flush_registered = False
        if not self._flush_registered:
            try:
                atexit.register(self._flush_persistence, True)
                self._flush_registered = True
            except Exception:
                pass
        self._set_status("memory", "loaded")

        # Try to load persisted embeddings from disk (keeps consistent with memory)
        self.embeddings_cache = load_embeddings()
        if self.embeddings_cache is not None:
            try:
                # Ensure cache is on CPU for consistent cos_sim calls
                self.embeddings_cache = self.embeddings_cache.to("cpu")
            except Exception:  # noqa: BLE001
                pass
            if getattr(self.embeddings_cache, "shape", (0,))[0] != len(
                self.corpus_texts
            ):
                clear_cache()
                self.embeddings_cache = None

        # If no cache, compute once (best-effort)
        if self.embeddings_cache is None and self.corpus_texts:
            try:
                self.embeddings_cache = self._encode(self.corpus_texts)
                save_embeddings(self.embeddings_cache)
                _report(0.65, "embeddings cached")
            except Exception:  # noqa: BLE001
                self.embeddings_cache = None

        # load research plugins (if any)
        from core.plugin_manager import load_plugins

        plugin_dir = os.environ.get("EVOAI_PLUGIN_DIR", "plugins")
        self.plugins = load_plugins(plugin_dir)
        _report(0.80, "plugins loaded")
        self._set_status("plugins", "loaded")

        # optionally load a text-generation LLM for richer replies
        self.llm_model = None
        self.llm_tokenizer = None
        self.llm_device = device
        llm_candidates: List[str] = []
        llm_primary = os.environ.get("EVOAI_LLM_MODEL", "").strip()
        if llm_primary:
            llm_candidates.append(llm_primary)
        llm_candidates.extend(_path_candidates(os.environ.get("EVOAI_LLM_MODEL_CANDIDATES")))
        if auto_discovery:
            llm_candidates.extend(
                [
                    os.path.join("data", "llm_finetuned"),
                    os.path.join("data", "llm_finetuned_debug"),
                ]
            )

        for cand in llm_candidates:
            llm_name = os.path.abspath(cand) if os.path.isdir(cand) else cand
            if not llm_name:
                continue
            try:
                # allow a simple "dummy" value for tests or stubs
                if llm_name == "dummy":
                    import types

                    # dummy model simply returns a fixed token stream; tests can
                    # override the tokenizer/model attributes directly if needed
                    class _DummyModel:
                        def generate(self, **kwargs):
                            # return a tensor of shape (1,1) for simplicity
                            if torch is not None:
                                return torch.tensor([[0]])
                            return [[0]]

                    self.llm_model = _DummyModel()
                    self.llm_tokenizer = types.SimpleNamespace(
                        __call__=lambda prompt, return_tensors=None: {"input_ids": torch.tensor([[0]]) if torch is not None else [[0]]},
                        decode=lambda ids, skip_special_tokens=True: ""
                    )
                    self._set_status("llm_selected", "dummy")
                    break
                else:
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
                    self.llm_model = AutoModelForCausalLM.from_pretrained(llm_name).to(device)
                    self._set_status("llm_selected", str(llm_name))
                    break
            except Exception:
                # if loading fails, just operate without LLM
                self.llm_model = None
                self.llm_tokenizer = None
                continue

        _report(0.95, "llm init done")

        # Optional external backend (GitHub Models by default).
        self.external_backend = None
        backend_provider = os.environ.get("EVOAI_BACKEND_PROVIDER", "github").strip().lower()
        self._set_status("backend_provider", backend_provider)
        self._set_status("backend_active", "false")
        self._set_status(
            "copilot_chat_enabled",
            "true" if os.environ.get("EVOAI_COPILOT_CHAT", "1").lower() in ("1", "true", "yes") else "false",
        )
        try:
            if backend_provider == "github":
                if os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"):
                    try:
                        self.external_backend = GitHubBackend()
                        self._set_status("backend_active", "true")
                    except Exception:
                        self.external_backend = None
        except Exception:
            self.external_backend = None

        # choose responder implementation
        responder_mode = os.environ.get("EVOAI_RESPONDER", "simple").lower()
        if responder_mode in ("smart", "auto"):
            self.responder = SmartResponder()
        else:
            self.responder = SimpleResponder()
        self._set_status("responder", responder_mode)

        _startup_log("Model loaded.")
        _report(1.0, "ready")
        self._set_status("ready", "yes")

        # Clear brain activity log file for fresh start
        try:
            activity_file = os.path.join(os.getcwd(), "data", "brain_activity.log")
            if os.path.exists(activity_file):
                os.remove(activity_file)
        except Exception:
            pass

        # optionally start REST API
        if os.environ.get("EVOAI_ENABLE_API", "").lower() in ("1", "true", "yes"):
            try:
                from core.api_server import run_server

                addr = os.environ.get("EVOAI_API_ADDR", "127.0.0.1")
                port = int(os.environ.get("EVOAI_API_PORT", "8000"))
                run_server(self, addr=addr, port=port)
                self._set_status("api", "running")
            except Exception:
                self._set_status("api", "error")

        # check for optional auto-update URL
        update_url = os.environ.get("EVOAI_AUTO_UPDATE_URL", "").strip()
        if update_url:
            try:
                from core.auto_updater import safe_run_update

                safe_run_update(update_url)
            except Exception:
                pass

    def _build_external_prompt(self, text: str) -> str:
        recent = self.memory[-8:]
        parts = ["You are EvoAI, a helpful assistant. Be clear, practical, and context-aware."]
        for entry in recent:
            if not isinstance(entry, dict):
                continue
            if entry.get("user"):
                parts.append(f"User: {entry.get('user')}")
            elif entry.get("assistant"):
                parts.append(f"Assistant: {entry.get('assistant')}")
        parts.append(f"User: {text}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _should_capture_for_training(self, user_text: str, reply: str) -> bool:
        user_words = len((user_text or "").strip().split())
        reply_words = len((reply or "").strip().split())
        if user_words < 3 or reply_words < 4:
            return False
        if (reply or "").lower().startswith("(error)"):
            return False
        if "no token provided" in (reply or "").lower():
            return False
        return True

    def _append_training_conversation(self, user_text: str, reply: str) -> None:
        if not self._should_capture_for_training(user_text, reply):
            return

        data_path = os.environ.get("EVOAI_TRAINING_DATA_PATH", os.path.join("data", "custom_conversations.json"))
        meta_path = os.environ.get(
            "EVOAI_TRAINING_META_PATH",
            os.path.join("data", "conversation_capture_meta.json"),
        )
        max_rows = int(os.environ.get("EVOAI_TRAINING_MAX_ROWS", "5000"))

        rows = getattr(self, "_training_rows_cache", None)
        if rows is None:
            rows = []
            try:
                if os.path.exists(data_path):
                    with open(data_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    if isinstance(existing, list):
                        rows = existing
            except Exception:
                rows = []
            self._training_rows_cache = rows
        elif not isinstance(rows, list):
            rows = []
            self._training_rows_cache = rows

        rows.append([str(user_text), str(reply)])
        if max_rows > 0 and len(rows) > max_rows:
            rows[:] = rows[-max_rows:]

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        meta = getattr(self, "_training_meta_cache", None)
        if meta is None:
            meta = {
                "new_samples_since_train": 0,
                "total_captured": 0,
                "last_capture_ts": 0,
            }
            try:
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        existing_meta = json.load(f)
                    if isinstance(existing_meta, dict):
                        meta.update(existing_meta)
            except Exception:
                pass
            self._training_meta_cache = meta
        elif not isinstance(meta, dict):
            meta = {
                "new_samples_since_train": 0,
                "total_captured": 0,
                "last_capture_ts": 0,
            }
            self._training_meta_cache = meta

        meta["new_samples_since_train"] = int(meta.get("new_samples_since_train", 0)) + 1
        meta["total_captured"] = int(meta.get("total_captured", 0)) + 1
        meta["last_capture_ts"] = int(time.time())

        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _log_brain_activity(self, filename: str):
        """Log brain activity to file for IPC with brain monitor."""
        try:
            activity_file = os.path.join(os.getcwd(), "data", "brain_activity.log")
            os.makedirs(os.path.dirname(activity_file), exist_ok=True)
            with open(activity_file, 'a') as f:
                f.write(f"{time.time()}|{filename}\n")
                f.flush()  # Force write to disk immediately
        except Exception:
            pass

    def _try_external_conversation_reply(self, text: str) -> str | None:
        self._log_brain_activity("github_backend.py")
        if os.environ.get("EVOAI_COPILOT_CHAT", "1").lower() not in ("1", "true", "yes"):
            return None
        if getattr(self, "external_backend", None) is None:
            return None
        try:
            prompt = self._build_external_prompt(text)
            model_name = os.environ.get("GITHUB_MODEL")
            reply = self.external_backend.generate_sync(prompt, model=model_name)
            if reply and str(reply).strip():
                self._set_status("backend_last", "success")
                return str(reply).strip()
        except Exception as e:
            self._set_status("backend_last", f"error:{e}")
        return None

    def _is_self_test_request(self, text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        intents = (
            "test yourself",
            "self test",
            "self-test",
            "run diagnostics",
            "diagnostic",
            "health check",
            "check yourself",
            "check all functions",
            "test all functions",
            "test all features",
            "full check",
            "system check",
        )
        return any(token in lowered for token in intents)

    def _humanize_self_test_result(self, ok: bool, brief: str) -> str:
        fallback = (
            "Self-test complete. Everything looks healthy and functional."
            if ok
            else f"Self-test found issues: {brief}. I can help you troubleshoot step by step."
        )
        if os.environ.get("EVOAI_COPILOT_NATURALIZE", "1").lower() not in ("1", "true", "yes"):
            return fallback
        if getattr(self, "external_backend", None) is None:
            return fallback
        try:
            model_name = os.environ.get("GITHUB_MODEL")
            prompt = (
                "You are EvoAI. Write one short, human, supportive status update for a completed self-test. "
                f"Result ok={ok}. Summary: {brief}. Keep it under 35 words."
            )
            text = self.external_backend.generate_sync(prompt, model=model_name)
            if text and str(text).strip():
                return str(text).strip()
        except Exception:
            pass
        return fallback

    def run_self_test(self, progress_cb=None) -> str:
        from core.self_repair import SelfRepair

        run_full = os.environ.get("EVOAI_SELF_TEST_FULL", "0").lower() in ("1", "true", "yes")
        ok, out = SelfRepair.run_tests(
            progress_cb=progress_cb,
            mode="repair",
            include_pytest=run_full,
        )
        brief = "all checks passed" if ok else (str(out).strip().splitlines()[-1] if out else "unknown error")
        mode_text = "full suite" if run_full else "fast diagnostics"
        human = self._humanize_self_test_result(ok, f"{brief} ({mode_text})")
        return f"{human} (self-test: {mode_text})"

    def try_handle_autonomous_request(self, text: str, progress_cb=None) -> str | None:
        if not self._is_self_test_request(text):
            return None
        allowed, reason = self.safety_gate.evaluate("self_test", text, self)
        if not allowed:
            self._set_status("safety_last_action", "self_test")
            self._set_status("safety_last_result", f"blocked:{reason}")
            return "Safety policy blocked autonomous self-test execution for this request."
        reply = self.run_self_test(progress_cb=progress_cb)
        try:
            self.record_interaction(str(text), reply)
        except Exception:
            pass
        return reply

    def generate_stream(self, prompt: str, chunk_callback, chunk_size: int = 64, record_user_text: str | None = None):
        """Generate text in a background thread and call `chunk_callback(chunk, final)`
        with successive chunks. Implementation uses full-generation then
        emits the output in slices to provide a streaming UI experience.
        """
        # If an external backend is configured, prefer streaming from it.
        if getattr(self, "external_backend", None) is not None:
            try:
                model_name = os.environ.get("GITHUB_MODEL")
                if record_user_text is None:
                    return self.external_backend.generate_stream(prompt, chunk_callback, model=model_name)

                assembled = []

                def _wrapped_chunk(chunk, final):
                    if chunk:
                        assembled.append(chunk)
                    chunk_callback(chunk, final)
                    if final:
                        try:
                            final_text = "".join(assembled).strip()
                            if final_text:
                                self.record_interaction(record_user_text, final_text)
                        except Exception:
                            pass

                return self.external_backend.generate_stream(prompt, _wrapped_chunk, model=model_name)
            except Exception as e:
                try:
                    chunk_callback(f"(external backend error) {e}", True)
                except Exception:
                    pass
                return None

        import threading

        def _worker():
            try:
                # reuse SmartResponder's generator helper by constructing inputs
                inputs = self.llm_tokenizer(prompt, return_tensors="pt")
                try:
                    model_device = next(self.llm_model.parameters()).device
                except Exception:
                    model_device = getattr(self, "llm_device", "cpu")
                for k, v in inputs.items():
                    try:
                        inputs[k] = v.to(model_device)
                    except Exception:
                        pass

                gen_kwargs = {
                    "max_new_tokens": int(self.llm_params.get("max_new_tokens", 50)),
                    "temperature": float(self.llm_params.get("temperature", 1.0)),
                    "top_k": int(self.llm_params.get("top_k", 0)),
                    "top_p": float(self.llm_params.get("top_p", 1.0)),
                    "do_sample": float(self.llm_params.get("temperature", 1.0)) > 0,
                }
                extra = self.llm_params.get("extra_gen_kwargs", {})
                if isinstance(extra, dict):
                    gen_kwargs.update(extra)

                outputs = self.llm_model.generate(**inputs, **gen_kwargs)
                text = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove prompt prefix if present
                if text.startswith(prompt):
                    text = text[len(prompt) :]

                # Emit in chunks
                i = 0
                n = len(text)
                while i < n:
                    j = min(i + chunk_size, n)
                    chunk = text[i:j]
                    chunk_callback(chunk, False)
                    i = j
                # final callback
                chunk_callback("", True)
                # record interaction if requested
                try:
                    if record_user_text is not None:
                        # final text is the whole generated output
                        self.record_interaction(record_user_text, text)
                except Exception:
                    pass
            except Exception as e:
                # Report error as final chunk
                try:
                    chunk_callback(f"(generation error) {e}", True)
                except Exception:
                    pass

        t = threading.Thread(target=_worker, daemon=True, name="EvoAI-Gen")
        t.start()
        return t

    def _flush_persistence(self, force: bool = False) -> None:
        if not force and not (self._dirty_memory or self._dirty_embeddings):
            return

        now = time.monotonic()
        if not force:
            if (
                self._pending_persist_turns < self._persist_every
                and (now - self._last_persist_ts) < self._persist_max_delay_sec
            ):
                return

        memory_ok = not self._dirty_memory
        embeddings_ok = (
            not self._dirty_embeddings
            or self.embeddings_cache is None
        )

        if self._dirty_memory:
            try:
                save_memory(self.memory)
                memory_ok = True
            except Exception:
                pass

        if self._dirty_embeddings and self.embeddings_cache is not None:
            try:
                save_embeddings(self.embeddings_cache)
                embeddings_ok = True
            except Exception:
                pass

        if memory_ok and embeddings_ok:
            self._pending_persist_turns = 0
            self._dirty_memory = False
            self._dirty_embeddings = False
            self._last_persist_ts = time.monotonic()

    def record_interaction(self, user_text: str, reply: str) -> None:
        """Store a user/assistant turn and update embeddings.

        ``max_memory_entries`` is applied after both entries are added, so the
        limit refers to total stored messages (not turns).
        """
        # append both user and assistant entries so history can include either
        self.memory.append({"user": user_text})
        self.memory.append({"assistant": reply})

        dropped_user_entries = 0
        if self.max_memory_entries and len(self.memory) > self.max_memory_entries:
            over_by = len(self.memory) - self.max_memory_entries
            dropped_slice = self.memory[:over_by]
            dropped_user_entries = sum(
                1
                for entry in dropped_slice
                if isinstance(entry, dict) and "user" in entry
            )
        self.memory = prune_memory(self.memory, self.max_memory_entries)
        self._dirty_memory = True

        # update repetition trackers
        self.last_user = user_text
        self.last_reply = reply

        try:
            self._append_training_conversation(user_text, reply)
        except Exception:
            pass

        # keep corpus_texts aligned with user entries only
        if dropped_user_entries > 0:
            self.corpus_texts = self.corpus_texts[dropped_user_entries:]
            try:
                if self.embeddings_cache is not None:
                    if torch is not None and hasattr(self.embeddings_cache, "shape"):
                        self.embeddings_cache = self.embeddings_cache[dropped_user_entries:]
                    elif isinstance(self.embeddings_cache, list):
                        self.embeddings_cache = self.embeddings_cache[dropped_user_entries:]
            except Exception:
                self.embeddings_cache = None

        self.corpus_texts.append(user_text)
        try:
            emb = self._encode(user_text)
            if torch is not None and hasattr(emb, "dim"):
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                if self.embeddings_cache is None:
                    self.embeddings_cache = emb.cpu().detach()
                else:
                    self.embeddings_cache = torch.cat([self.embeddings_cache, emb.cpu().detach()], dim=0)
            else:
                if self.embeddings_cache is None:
                    self.embeddings_cache = [emb]
                else:
                    self.embeddings_cache.append(emb)
            self._dirty_embeddings = True
        except Exception:  # noqa: BLE001
            try:
                clear_cache()
            except Exception:
                pass
            self.embeddings_cache = None
            self._dirty_embeddings = False

        self._pending_persist_turns += 1
        self._flush_persistence(force=False)

        # (moved out into __init__)
    def respond(self, text):
        # Log brain activity
        self._log_brain_activity("engine_template.py")
        
        # delegate to the configured responder implementation; this keeps the
        # logic centralized and makes the behaviour configurable via
        # ``EVOAI_RESPONDER``.  The responders themselves handle empty input
        # checks and memory updates.
        if not text or not str(text).strip():
            return "Please enter something."

        # Handle special commands
        text_lower = str(text).strip().lower()
        
        # Backend control command
        if text_lower == "disable backend":
            if self.external_backend is None:
                return "Backend is already disabled. Using local processing only."
            else:
                self.external_backend = None
                self._set_status("backend_active", "false")
                return "✓ Backend disabled. Now using local (non-API) processing. Backend can be re-enabled by restarting."
        
        if text_lower == "enable backend":
            if self.external_backend is not None:
                return "Backend is already enabled."
            else:
                # Re-enable backend
                try:
                    from core.github_backend import GitHubBackend
                    if os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"):
                        self.external_backend = GitHubBackend()
                        self._set_status("backend_active", "true")
                        return "✓ Backend re-enabled. Using GitHub Models API."
                    else:
                        return "Cannot enable backend: No GitHub token available. Set GITHUB_TOKEN environment variable."
                except Exception as e:
                    return f"Could not re-enable backend: {e}"
        
        if text_lower == "backend status":
            status = "enabled (GitHub Models API)" if self.external_backend else "disabled (local processing)"
            return f"Backend status: {status}"

        autonomous = self.try_handle_autonomous_request(str(text), progress_cb=None)
        if autonomous is not None:
            return autonomous

        # When backend is disabled, use local responder for normal conversation
        if self.external_backend is None:
            self._log_brain_activity("memory.py")
            reply = self.responder.respond(str(text), self)
            self._turns_since_proactive += 1
            
            # maybe ask for a clarification if the user input seemed vague
            try:
                from core import language_utils
                self._log_brain_activity("language_utils.py")

                clar, topic = language_utils.clarify_if_ambiguous(text)
                if clar and topic and topic not in self.clarifications_asked:
                    self.clarifications_asked.add(topic)
                    reply = clar + " " + reply
            except Exception:
                pass

            # optionally enhance vocabulary of the reply
            if os.environ.get("EVOAI_USE_THESAURUS", "1").lower() in ("1", "true", "yes"):
                try:
                    from core import language_utils
                    self._log_brain_activity("language_utils.py")

                    reply = language_utils.enhance_text(reply)
                except Exception:
                    pass
            
            return reply

        ext_allowed, ext_reason = self.safety_gate.evaluate("external_backend", str(text), self)
        external_reply = self._try_external_conversation_reply(str(text)) if ext_allowed else None
        if not ext_allowed:
            self._set_status("safety_last_action", "external_backend")
            self._set_status("safety_last_result", f"blocked:{ext_reason}")
        if external_reply:
            self.record_interaction(str(text), external_reply)
            self._set_status("decision_last_action", "external_backend")
            self._set_status("decision_last_latency_ms", "0.00")
            self._set_status("decision_last_reason", "github_models")
            reply = external_reply
            try:
                from core import language_utils

                clar, topic = language_utils.clarify_if_ambiguous(text)
                if clar and topic and topic not in self.clarifications_asked:
                    self.clarifications_asked.add(topic)
                    reply = clar + " " + reply
            except Exception:
                pass

            if os.environ.get("EVOAI_USE_THESAURUS", "1").lower() in ("1", "true", "yes"):
                try:
                    from core import language_utils

                    reply = language_utils.enhance_text(reply)
                except Exception:
                    pass
            return reply

        decision = {
            "action": "delegate",
            "confidence": 1.0,
            "elapsed_ms": 0.0,
            "reason": "default",
        }
        try:
            self._log_brain_activity("decision_policy.py")
            decision = self.decision_policy.decide(text, self)
        except Exception:
            decision = {
                "action": "delegate",
                "confidence": 0.0,
                "elapsed_ms": 0.0,
                "reason": "error",
            }

        action = decision.get("action", "delegate")
        self._set_status("decision_last_action", str(action))
        self._set_status("decision_last_latency_ms", f"{decision.get('elapsed_ms', 0.0):.2f}")
        self._set_status("decision_last_reason", str(decision.get("reason", "unknown")))

        gov_allowed, gov_reason = self._governance_allows(str(action))
        if not gov_allowed:
            self._set_status("safety_last_action", str(action))
            self._set_status("safety_last_result", f"blocked:{gov_reason}")
            self._audit_event("blocked", str(action), gov_reason, text)
            reply = "Autonomy governance blocked this action, so I’ll continue in safe mode."
            safe_reply = self.responder.respond(text, self)
            return f"{reply} {safe_reply}".strip()

        allowed, reason = self.safety_gate.evaluate(str(action), str(text), self)
        previous_safety_result = str(self._status.get("safety_last_result", ""))
        previous_was_blocked = previous_safety_result.startswith("blocked:")
        if not previous_was_blocked or not allowed:
            self._set_status("safety_last_action", str(action))
            self._set_status("safety_last_result", "ok" if allowed else f"blocked:{reason}")
        if not allowed:
            self._audit_event("blocked", str(action), reason, text)
            reply = "I can’t run that autonomous action under current safety settings, so I’ll continue in safe mode."
            safe_reply = self.responder.respond(text, self)
            return f"{reply} {safe_reply}".strip()

        self._audit_event("allow", str(action), "ok", text)

        if action == "simple_reply":
            self._log_brain_activity("memory.py")
            reply = SimpleResponder().respond(text, self)
        elif action == "llm_generate" and self.llm_model and self.llm_tokenizer:
            self._log_brain_activity("trainer.py")
            reply = SmartResponder().generate_with_model(
                SmartResponder().build_prompt(text, self),
                self,
            )
            if not reply:
                reply = self.responder.respond(text, self)
            else:
                self.record_interaction(text, reply)
        elif action == "proactive_prompt":
            reply = self.responder.respond(text, self)
            if (
                self.decision_policy.allow_autonomy
                and self._turns_since_proactive >= self.decision_autonomy_cooldown
            ):
                extra = " What would you like me to proactively focus on next?"
                reply = (reply or "").strip() + extra
                self._turns_since_proactive = 0
            else:
                self._turns_since_proactive += 1
        elif action == "clarify_first":
            self._log_brain_activity("language_utils.py")
            try:
                from core import language_utils

                clar, _topic = language_utils.clarify_if_ambiguous(text)
                base = self.responder.respond(text, self)
                reply = (clar + " " + base).strip() if clar else base
            except Exception:
                reply = self.responder.respond(text, self)
        elif action == "autonomy_plan":
            reply = (
                "Autonomy plan scaffold is active: define objective, enforce safety budget, "
                "then execute incremental validated actions."
            )
            self.record_interaction(text, reply)
        elif action == "safety_check":
            meta = self.safety_gate.metadata()
            reply = (
                "Safety gate status — "
                f"enabled={meta.get('enabled')}, "
                f"network={meta.get('allow_network_actions')}, "
                f"self_modify={meta.get('allow_self_modify')}, "
                f"autonomy={meta.get('allow_autonomy_actions')}"
            )
            self.record_interaction(text, reply)
        elif action == "code_intel_query":
            self._log_brain_activity("plugin_manager.py")
            report = self.code_intel_toolkit.analyze(text)
            reply = str(report.get("summary", "Code intel query completed."))
            self._set_status("code_intel_last_matches", str(len(report.get("matches", []))))
            self.record_interaction(text, reply)
        elif action == "research_query":
            self._log_brain_activity("network_scanner.py")
            result = self.research_toolkit.research(text, self)
            reply = str(result.get("summary", "Research query completed."))
            self._set_status("research_last_plugin_hits", str(len(result.get("plugin_findings", []))))
            self._set_status("research_last_web_results", str(len(result.get("web_results", []))))
            self.record_interaction(text, reply)
        elif action == "tested_apply":
            self._log_brain_activity("self_repair.py")
            outcome = self.tested_apply_orchestrator.run(text)
            reply = str(outcome.get("summary", "Tested apply finished."))
            self._set_status("tested_apply_last_ok", str(bool(outcome.get("ok", False))).lower())
            self._set_status("tested_apply_last_reason", str(outcome.get("reason", "unknown")))
            self._set_status("tested_apply_last_files", str(int(outcome.get("files", 0))))
            self._set_status("tested_apply_last_score", f"{float(outcome.get('retention_score', 0.0)):.3f}")
            self._audit_event(
                "outcome",
                "tested_apply",
                str(outcome.get("reason", "unknown")),
                text,
                meta={
                    "ok": bool(outcome.get("ok", False)),
                    "files": int(outcome.get("files", 0)),
                    "retention_score": float(outcome.get("retention_score", 0.0) or 0.0),
                },
            )
            self.record_interaction(text, reply)
        elif action == "code_assist":
            workflow_result = self.code_assistant.workflow(text, engine=self)
            if workflow_result.get("ok"):
                step = workflow_result.get("step", "unknown")
                if step == "validation_complete" or step == "validation_blocked":
                    candidate = workflow_result.get("candidate", "")
                    reply = f"Code analysis complete. Candidate code generated:\n\n{candidate}\n\nReview needed before applying."
                    self._set_status("code_assist_step", "validation_complete")
                elif step == "apply":
                    apply_result = workflow_result.get("apply_result", {})
                    reply = f"Code applied successfully. Retention score: {apply_result.get('retention_score', 0.0):.3f}"
                    self._set_status("code_assist_step", "applied")
                else:
                    reply = f"Code assist completed at step: {step}"
                    self._set_status("code_assist_step", step)
            else:
                step = workflow_result.get("step", "unknown")
                reason = workflow_result.get("reason", "unknown")
                reply = f"Code assist stopped at {step}: {reason}"
                self._set_status("code_assist_step", f"failed:{step}")
            
            self._set_status("code_assist_last_reason", workflow_result.get("reason", "unknown"))
            analysis = workflow_result.get("analysis", {})
            self._set_status("code_assist_matches", str(len(analysis.get("matches", []))))
            self._audit_event("outcome", "code_assist", workflow_result.get("reason", ""), text)
            self.record_interaction(text, reply)
        else:
            reply = self.responder.respond(text, self)
            self._turns_since_proactive += 1

        # maybe ask for a clarification if the user input seemed vague
        try:
            from core import language_utils

            clar, topic = language_utils.clarify_if_ambiguous(text)
            if (
                action != "clarify_first"
                and clar
                and topic
                and topic not in self.clarifications_asked
            ):
                self.clarifications_asked.add(topic)
                reply = clar + " " + reply
        except Exception:
            pass

        # optionally enhance vocabulary of the reply
        if os.environ.get("EVOAI_USE_THESAURUS", "1").lower() in ("1", "true", "yes"):
            try:
                from core import language_utils

                reply = language_utils.enhance_text(reply)
            except Exception:
                pass

        return reply



    def status(self) -> dict:
        """Return a snapshot of the engine's internal status dictionary."""
        return dict(self._status)

    def _apply_governance_policy_defaults(self) -> None:
        self.governance_policy_loaded = False
        self.governance_policy_recommendation = "none"
        self.governance_policy_applied = "none"

        autotune = os.environ.get("EVOAI_GOVERNANCE_POLICY_AUTOTUNE", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        if not autotune:
            self.governance_policy_applied = "autotune_disabled"
            return

        policy_path = os.environ.get(
            "EVOAI_GOVERNANCE_POLICY_PATH",
            os.path.join("data", "governance_policy", "metadata.json"),
        )
        if not os.path.exists(policy_path):
            self.governance_policy_applied = "policy_missing"
            return

        try:
            with open(policy_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception:
            self.governance_policy_applied = "policy_invalid"
            return

        if not isinstance(metadata, dict):
            self.governance_policy_applied = "policy_invalid"
            return

        recommendation = str(metadata.get("recommendation", "balanced") or "balanced").strip().lower()
        self.governance_policy_loaded = True
        self.governance_policy_recommendation = recommendation

        if "EVOAI_AUTONOMY_BUDGET_MAX" in os.environ:
            self.governance_policy_applied = "skipped_env_override"
            return

        before_max = int(self.autonomy_budget_max)
        if recommendation == "tighten_autonomy":
            tuned = max(1, int(round(before_max * 0.5)))
            self.autonomy_budget_max = tuned
            self.autonomy_budget_remaining = min(int(self.autonomy_budget_remaining), tuned)
            self.governance_policy_applied = "budget_tightened"
        elif recommendation == "loosen_autonomy":
            tuned = min(500, max(before_max + 1, int(round(before_max * 1.5))))
            self.autonomy_budget_max = tuned
            self.autonomy_budget_remaining = min(int(self.autonomy_budget_remaining), tuned)
            self.governance_policy_applied = "budget_loosened"
        else:
            self.governance_policy_applied = "balanced_no_change"

    def governance_status(self) -> dict:
        return {
            "autonomy_paused": bool(self.autonomy_paused),
            "autonomy_budget_max": int(self.autonomy_budget_max),
            "autonomy_budget_remaining": int(self.autonomy_budget_remaining),
            "audit_events": len(self._audit_events),
            "governance_policy_loaded": bool(self.governance_policy_loaded),
            "governance_policy_recommendation": str(self.governance_policy_recommendation),
            "governance_policy_applied": str(self.governance_policy_applied),
        }

    def update_governance(self, payload: dict | None) -> dict:
        body = payload if isinstance(payload, dict) else {}
        if "autonomy_paused" in body:
            self.autonomy_paused = bool(body.get("autonomy_paused"))

        if "autonomy_budget_max" in body:
            try:
                self.autonomy_budget_max = max(0, int(body.get("autonomy_budget_max")))
            except Exception:
                pass

        if body.get("reset_budget"):
            self.autonomy_budget_remaining = int(self.autonomy_budget_max)

        if "autonomy_budget_remaining" in body:
            try:
                value = max(0, int(body.get("autonomy_budget_remaining")))
                self.autonomy_budget_remaining = min(value, int(self.autonomy_budget_max))
            except Exception:
                pass

        self._set_status("autonomy_paused", str(self.autonomy_paused).lower())
        self._set_status("autonomy_budget_max", str(self.autonomy_budget_max))
        self._set_status("autonomy_budget_remaining", str(self.autonomy_budget_remaining))
        return self.governance_status()

    def audit_events(self, limit: int = 50) -> list[dict]:
        lim = max(1, int(limit))
        items = list(self._audit_events)
        return items[-lim:]

    def _audit_event(self, kind: str, action: str, reason: str, text: str, meta: dict | None = None) -> None:
        try:
            event = {
                "ts": int(time.time()),
                "kind": str(kind),
                "action": str(action),
                "reason": str(reason),
                "query": str(text or "")[:180],
            }
            if isinstance(meta, dict) and meta:
                event["meta"] = meta
            self._audit_events.append(event)
            self._set_status("audit_events", str(len(self._audit_events)))
            if self.outcome_log_enabled:
                out_path = str(self.outcomes_path)
                parent = os.path.dirname(out_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(out_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _governance_allows(self, action: str) -> tuple[bool, str]:
        action_name = str(action or "delegate")
        autonomy_actions = getattr(self.safety_gate, "AUTONOMY_ACTIONS", set())
        if action_name not in autonomy_actions:
            return True, "ok"
        if self.autonomy_paused:
            return False, "autonomy_paused"
        if self.autonomy_budget_remaining <= 0:
            return False, "autonomy_budget_exhausted"
        self.autonomy_budget_remaining -= 1
        self._set_status("autonomy_budget_remaining", str(self.autonomy_budget_remaining))
        return True, "ok"

    def _set_status(self, key: str, value: str) -> None:
        try:
            self._status[key] = value
        except Exception:
            pass


def handle_exit(sig, frame):
    print("\nShutting down EvoAI safely...")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_exit)
