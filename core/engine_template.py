import os
import signal
import sys
import torch
from sentence_transformers import SentenceTransformer, util
from core.memory import load_memory, save_memory
from core.embeddings_cache import load_embeddings, save_embeddings, clear_cache
from core.openai_backend import OpenAIBackend


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
                    return (
                        f"I recall you mentioned '{previous}' earlier. "
                        "Could you tell me more?"
                    )
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
        print("Loading model (Intel CPU optimized)...")
        torch.set_num_threads(int(os.environ.get("EVOAI_TORCH_THREADS", "8")))

        # allow previously fine-tuned model to override the base name
        model_name = os.environ.get("EVOAI_FINETUNED_MODEL") or os.environ.get("EVOAI_MODEL", "all-MiniLM-L6-v2")
        # Force CPU device to maintain determinism on Intel Mac
        device = os.environ.get("EVOAI_DEVICE", "cpu")

        self._set_status("config", "loaded")

        _report(0.10, "loading embeddings model")
        self.model = SentenceTransformer(model_name, device=device)
        _report(0.45, "embeddings model loaded")
        self._set_status("embeddings_model", "ok")
        # small optimization: cache encode method with convert_to_tensor=True
        self._encode = lambda texts, **k: self.model.encode(
            texts, convert_to_tensor=True, **k
        )

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
        llm_name = os.environ.get("EVOAI_LLM_MODEL", "").strip()
        if llm_name:
            try:
                # allow a simple "dummy" value for tests or stubs
                if llm_name == "dummy":
                    import types

                    # dummy model simply returns a fixed token stream; tests can
                    # override the tokenizer/model attributes directly if needed
                    class _DummyModel:
                        def generate(self, **kwargs):
                            # return a tensor of shape (1,1) for simplicity
                            return torch.tensor([[0]])

                    self.llm_model = _DummyModel()
                    self.llm_tokenizer = types.SimpleNamespace(
                        __call__=lambda prompt, return_tensors=None: {"input_ids": torch.tensor([[0]])},
                        decode=lambda ids, skip_special_tokens=True: ""
                    )
                else:
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
                    self.llm_model = AutoModelForCausalLM.from_pretrained(llm_name).to(device)
            except Exception:
                # if loading fails, just operate without LLM
                self.llm_model = None
                self.llm_tokenizer = None

                _report(0.95, "llm init done")

        # Optional OpenAI ChatGPT backend (used as "backend thinking")
        self.openai_backend = None
        try:
            if os.environ.get("OPENAI_API_KEY"):
                try:
                    self.openai_backend = OpenAIBackend()
                except Exception:
                    # don't fail construction if OpenAI isn't usable
                    self.openai_backend = None
        except Exception:
            self.openai_backend = None

        # choose responder implementation
        responder_mode = os.environ.get("EVOAI_RESPONDER", "simple").lower()
        if responder_mode in ("smart", "auto"):
            self.responder = SmartResponder()
        else:
            self.responder = SimpleResponder()
        self._set_status("responder", responder_mode)

        print("Model loaded.")
        _report(1.0, "ready")
        self._set_status("ready", "yes")

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

        # optional network awareness stub (requires explicit user consent)
        if os.environ.get("EVOAI_ENABLE_NET_SCAN", "").lower() in ("1", "true", "yes", "permitted"):
            try:
                from core.network_scanner import scan_local_network

                scan_local_network()
                self._set_status("network_scan", "noop")
            except Exception:
                self._set_status("network_scan", "error")

        # check for optional auto-update URL
        update_url = os.environ.get("EVOAI_AUTO_UPDATE_URL", "").strip()
        if update_url:
            try:
                from core.auto_updater import safe_run_update

                safe_run_update(update_url)
            except Exception:
                pass

    def generate_stream(self, prompt: str, chunk_callback, chunk_size: int = 64, record_user_text: str | None = None):
        """Generate text in a background thread and call `chunk_callback(chunk, final)`
        with successive chunks. Implementation uses full-generation then
        emits the output in slices to provide a streaming UI experience.
        """
        # If an OpenAI backend is configured, prefer streaming from it
        if getattr(self, "openai_backend", None) is not None:
            try:
                return self.openai_backend.generate_stream(prompt, chunk_callback, model=os.environ.get("OPENAI_MODEL"))
            except Exception as e:
                try:
                    chunk_callback(f"(openai backend error) {e}", True)
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

    def record_interaction(self, user_text: str, reply: str) -> None:
        """Store a user/assistant turn and update embeddings.

        ``max_memory_entries`` is applied after both entries are added, so the
        limit refers to total stored messages (not turns).
        """
        # append both user and assistant entries so history can include either
        self.memory.append({"user": user_text})
        self.memory.append({"assistant": reply})
        save_memory(self.memory, max_entries=self.max_memory_entries)

        # update repetition trackers
        self.last_user = user_text
        self.last_reply = reply

        # keep corpus_texts aligned with user entries only
        self.corpus_texts.append(user_text)
        try:
            emb = self._encode(user_text)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            if self.embeddings_cache is None:
                self.embeddings_cache = emb.cpu().detach()
            else:
                self.embeddings_cache = torch.cat([self.embeddings_cache, emb.cpu().detach()], dim=0)
            save_embeddings(self.embeddings_cache)
        except Exception:  # noqa: BLE001
            try:
                clear_cache()
            except Exception:
                pass
            self.embeddings_cache = None

        # (moved out into __init__)
    def respond(self, text):
        # delegate to the configured responder implementation; this keeps the
        # logic centralized and makes the behaviour configurable via
        # ``EVOAI_RESPONDER``.  The responders themselves handle empty input
        # checks and memory updates.
        reply = self.responder.respond(text, self)

        # maybe ask for a clarification if the user input seemed vague
        try:
            from core import language_utils

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

                reply = language_utils.enhance_text(reply)
            except Exception:
                pass

        return reply



    def status(self) -> dict:
        """Return a snapshot of the engine's internal status dictionary."""
        return dict(self._status)

    def _set_status(self, key: str, value: str) -> None:
        try:
            self._status[key] = value
        except Exception:
            pass


def handle_exit(sig, frame):
    print("\nShutting down EvoAI safely...")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_exit)
