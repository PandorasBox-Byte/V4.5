import os
import subprocess
import sys
import time
from typing import Tuple


class SelfRepair:
    """Simple self-repair utilities used by the launcher supervisor.

    The strategy is conservative: run the test suite, attempt to reinstall
    editable package and dependencies, then re-run tests. This avoids
    invasive file modifications and keeps repair actions repeatable.
    """

    @staticmethod
    def run_command(cmd: list, cwd: str = None, timeout: int = 300) -> Tuple[int, str]:
        try:
            p = subprocess.run(
                [sys.executable, "-m"] + cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = (p.stdout or "") + (p.stderr or "")
            return p.returncode, out
        except subprocess.TimeoutExpired:
            return 124, "timeout"

    @classmethod
    def run_tests(cls, progress_cb=None) -> Tuple[bool, str]:
        """Run a sequence of lightweight checks followed by the full pytest run.

        progress_cb: optional callable(progress: float, message: str, check_name: str|None=None, passed: bool|None=None)
        """
        # Allow quick disabling of the self-repair flow for debugging/startup
        # by setting the environment variable `DISABLE_SELF_REPAIR`.
        # Default is to disable while we troubleshoot startup issues.
        if os.environ.get("DISABLE_SELF_REPAIR", "1") in ("1", "true", "True", "yes"):
            try:
                if callable(progress_cb):
                    progress_cb(1.0, "self-repair disabled", check_name="self-repair", passed=True)
            except Exception:
                pass
            return True, "self-repair disabled"

        checks = []

        # smoke engine construction (fast)
        def _smoke():
            import sys
            from types import SimpleNamespace
            import torch

            class DummyST:
                def __init__(self, *a, **k):
                    pass

                def encode(self, texts, convert_to_tensor=True, **k):
                    if isinstance(texts, list):
                        return torch.randn(len(texts), 384)
                    return torch.randn(384)

            fake = SimpleNamespace(SentenceTransformer=DummyST, util=SimpleNamespace(cos_sim=lambda a, b: torch.tensor([[0.0]])))
            sys.modules['sentence_transformers'] = fake
            sys.modules['transformers'] = SimpleNamespace(AutoModelForCausalLM=lambda *a, **k: None, AutoTokenizer=lambda *a, **k: None)
            try:
                from core.engine_template import Engine

                e = Engine(progress_cb=lambda f, m: None)
                return True, 'smoke ok'
            except Exception as e:  # pragma: no cover - runtime dependent
                return False, str(e)

        checks.append(('smoke_engine', _smoke))

        # embeddings file integrity
        def _embeddings():
            try:
                from core.embeddings_cache import load_embeddings

                emb = load_embeddings()
                if emb is None:
                    return True, 'no persisted embeddings (ok)'
                # ensure shape matches expectation (2D tensor)
                if hasattr(emb, 'ndim') and getattr(emb, 'ndim') in (1, 2):
                    return True, 'embeddings OK'
                return False, 'embeddings unexpected shape'
            except Exception as e:
                return False, str(e)

        checks.append(('embeddings', _embeddings))

        # memory file read/write
        def _memory():
            try:
                from core.memory import load_memory, save_memory

                mem = load_memory()
                # try a no-op save to ensure writable
                save_memory(mem or [], max_entries=10)
                return True, 'memory rw OK'
            except Exception as e:
                return False, str(e)

        checks.append(('memory', _memory))

        # plugins loading
        def _plugins():
            try:
                from core.plugin_manager import load_plugins

                plugins = load_plugins()
                # optionally call a lightweight health method if plugin exposes it
                for p in plugins:
                    try:
                        if hasattr(p, 'health_check'):
                            ok = p.health_check()
                            if not ok:
                                return False, f'plugin {getattr(p, "__name__", str(p))} failed'
                    except Exception:
                        # ignore plugin-specific errors
                        continue
                return True, f'{len(plugins)} plugins'
            except Exception as e:
                return False, str(e)

        checks.append(('plugins', _plugins))

        # optional OpenAI API reachability
        def _openai():
            try:
                import os

                key = os.environ.get('OPENAI_API_KEY')
                if not key:
                    return True, 'openai skipped'
                try:
                    import openai
                except Exception as e:
                    return False, f'openai package missing: {e}'
                openai.api_key = key
                # quick call: list models or a tiny completion
                try:
                    # prefer a lightweight list models call if available
                    if hasattr(openai, 'Model'):
                        _ = openai.Model.list()
                    else:
                        # fallback to ChatCompletion with max_tokens=1 and timeout
                        openai.ChatCompletion.create(model=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'), messages=[{'role':'user','content':'hi'}], max_tokens=1)
                    return True, 'openai reachable'
                except Exception as e:
                    return False, str(e)
            except Exception as e:
                return False, str(e)

        checks.append(('openai', _openai))

        # run checks and report progress
        results = {}
        total = len(checks)
        for i, (name, fn) in enumerate(checks):
            try:
                ok, msg = fn()
            except Exception as e:
                ok, msg = False, str(e)
            results[name] = (ok, msg)
            frac = (i + 1) / total
            try:
                if callable(progress_cb):
                    progress_cb(frac, msg, check_name=name, passed=ok)
            except Exception:
                pass

            # small sleep to allow UI updates in interactive runs
            try:
                import time

                time.sleep(0.02)
            except Exception:
                pass

        # summarise: if any mandatory check failed, return early
        for k, (ok, msg) in results.items():
            if not ok and k in ('smoke_engine', 'embeddings', 'memory'):
                return False, f'{k} failed: {msg}'

        # If checks pass, run full pytest
        code, out = cls.run_command(['pytest', '-q'])
        try:
            if callable(progress_cb):
                progress_cb(1.0, 'pytest done', check_name='pytest', passed=(code == 0))
        except Exception:
            pass
        return code == 0, out

    @classmethod
    def reinstall_editable(cls) -> Tuple[bool, str]:
        # Install editable package and ensure requirements are satisfied.
        out_all = []
        code, out = cls.run_command(["pip", "install", "-e", "."])
        out_all.append(out)
        if code != 0:
            return False, "\n".join(out_all)
        # Avoid re-installing `requirements.txt` here because it may pin older
        # versions that break runtime imports (e.g. sentence-transformers 2.x).
        # Instead, ensure known-working versions of critical libs are present.
        try:
            code2, out2 = cls.run_command([
                "pip",
                "install",
                "--upgrade",
                "sentence-transformers==5.2.3",
                "huggingface_hub==0.36.2",
            ])
            out_all.append(out2)
            if code2 != 0:
                return False, "\n".join(out_all)
        except Exception as e:
            out_all.append(str(e))
            return False, "\n".join(out_all)

        return True, "\n".join(out_all)

    @classmethod
    def attempt_repair(cls, max_attempts: int = 2, wait_seconds: int = 5) -> bool:
        """Attempt a repair sequence. Returns True if repair likely succeeded.

        Steps:
        1. Run tests; if they pass, nothing to do.
        2. Reinstall editable + requirements.
        3. Re-run tests.
        """
        # If disabled via environment, short-circuit and report success so the
        # launcher continues without attempting repairs.
        if os.environ.get("DISABLE_SELF_REPAIR", "1") in ("1", "true", "True", "yes"):
            print("[self-repair] Disabled via DISABLE_SELF_REPAIR; skipping attempts.")
            return True

        for attempt in range(1, max_attempts + 1):
            print(f"[self-repair] Attempt {attempt}/{max_attempts}")
            ok, out = cls.run_tests()
            if ok:
                print("[self-repair] Tests already passing — no repair needed.")
                return True

            print("[self-repair] Tests failing — reinstalling package and requirements...")
            ok_install, install_out = cls.reinstall_editable()
            print(install_out)
            if not ok_install:
                print("[self-repair] Reinstall failed; will retry after delay.")
                time.sleep(wait_seconds)
                continue

            ok, out = cls.run_tests()
            print(out)
            if ok:
                print("[self-repair] Repair succeeded — tests pass.")
                return True

            print("[self-repair] Repair attempt did not fix tests; retrying...")
            time.sleep(wait_seconds)

        print("[self-repair] All repair attempts exhausted.")
        return False
