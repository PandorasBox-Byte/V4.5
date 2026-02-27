import os
import signal
import sys
import time
import json
import subprocess
import io
import contextlib
import warnings

# When this file is executed as a script (python core/launcher.py),
# Python sets sys.path[0] to the script's directory (core/). That makes
# `import core...` fail because the package root isn't on sys.path.
# Insert the project root into sys.path so `import core.engine_template`
# works whether launched as a module or as a script.
if __name__ == "__main__" and __package__ is None:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_this_dir)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

from core.engine_template import Engine
from core.self_repair import SelfRepair
from core import tui

PIDFILE = os.path.join("data", "engine.pid")


def write_pidfile():
    os.makedirs(os.path.dirname(PIDFILE), exist_ok=True)
    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))


def remove_pidfile():
    try:
        if os.path.exists(PIDFILE):
            os.remove(PIDFILE)
    except Exception:
        pass


def handle_exit(sig, frame):
    print("\nShutting down EvoAI (launcher cleanup)...")
    remove_pidfile()
    sys.exit(0)


# internal flag preventing repeated prompting
_prompted_for_key = False


def _maybe_run_startup_training() -> None:
    # Keep startup UX responsive by default; enable only when explicitly requested.
    if os.environ.get("EVOAI_STARTUP_AUTO_TRAIN", "0").lower() not in ("1", "true", "yes"):
        return

    meta_path = os.environ.get(
        "EVOAI_TRAINING_META_PATH",
        os.path.join("data", "conversation_capture_meta.json"),
    )
    threshold = int(os.environ.get("EVOAI_STARTUP_TRAIN_THRESHOLD", "80"))

    meta = {}
    try:
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                meta = loaded
    except Exception:
        return

    new_samples = int(meta.get("new_samples_since_train", 0))
    if new_samples < threshold:
        return

    train_script = os.path.join("scripts", "train_personalization.py")
    if not os.path.exists(train_script):
        return

    timeout_s = int(os.environ.get("EVOAI_STARTUP_TRAIN_TIMEOUT", "1200"))
    emb_epochs = os.environ.get("EVOAI_STARTUP_TRAIN_EMB_EPOCHS", "1")
    llm_epochs = os.environ.get("EVOAI_STARTUP_TRAIN_LLM_EPOCHS", "1")
    decision_epochs = os.environ.get("EVOAI_STARTUP_TRAIN_DECISION_EPOCHS", "10")

    print(f"[launcher] starting local training ({new_samples} new captured turns)")
    cmd = [
        sys.executable,
        train_script,
        "--emb-epochs",
        str(emb_epochs),
        "--llm-epochs",
        str(llm_epochs),
        "--decision-epochs",
        str(decision_epochs),
    ]
    try:
        completed = subprocess.run(cmd, check=False, timeout=timeout_s)
        if completed.returncode == 0:
            meta["new_samples_since_train"] = 0
            meta["last_train_ok"] = True
            meta["last_train_ts"] = int(time.time())
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print("[launcher] startup training complete")
        else:
            print(f"[launcher] startup training failed with code {completed.returncode}")
    except Exception as e:
        print(f"[launcher] startup training skipped: {e}")

def _maybe_prompt_for_api_key() -> None:
    """Prompt for a GitHub token during launcher startup.

    The prompt runs before engine/model loading so authenticated backend
    calls are available immediately in the first session turn.
    """
    global _prompted_for_key
    if _prompted_for_key:
        return

    if sys.stdout.isatty() and not (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")):
        try:
            import getpass

            key = getpass.getpass(
                "Paste your GitHub token (leave empty to skip, not saved): "
            )
            # strip whitespace; empty means we explicitly do not want to
            # use a key for this session.  Remove any existing (empty)
            # environment variable to avoid re-prompting later.
            if key and key.strip():
                os.environ["GITHUB_TOKEN"] = key.strip()
            else:
                # explicit user choice to skip; clear any existing variable
                os.environ.pop("GITHUB_TOKEN", None)
                os.environ.pop("GH_TOKEN", None)
                # let the user know the engine will continue without key
                print("[launcher] no token provided, continuing without GitHub backend access")
        except Exception:
            # non-fatal: continue without setting a key
            pass
    _prompted_for_key = True


def main():
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    write_pidfile()

    # Supervisor loop: prefer an ASCII TUI when run in a terminal.
    restart_delay = 2

    def _has_interactive_tty() -> bool:
        try:
            forced = os.environ.get("EVOAI_FORCE_TUI", "").strip().lower()
            if forced in ("1", "true", "yes"):
                return True
            if forced in ("0", "false", "no"):
                return False

            streams = (sys.stdin, sys.stdout, sys.stderr)
            if any(getattr(s, "isatty", lambda: False)() for s in streams):
                return True

            term = os.environ.get("TERM", "").strip().lower()
            if term and term not in ("dumb", "unknown"):
                return True
        except Exception:
            pass
        return False

    def _run_repl_loop(engine_obj):
        print("Engine started (no TTY) — using REPL.")
        while True:
            try:
                user_input = input("You: ")
            except EOFError:
                raise SystemExit
            if user_input is None:
                raise SystemExit
            if not user_input.strip():
                continue
            try:
                response = engine_obj.respond(user_input)
                print("EvoAI:", response)
            except Exception as e:
                print(f"[launcher] Engine error during respond: {e}")
                repaired = SelfRepair.attempt_repair()
                if repaired:
                    print("[launcher] Repair succeeded; restarting engine.")
                    break
                else:
                    print("[launcher] Repair failed; will retry after delay.")
                    time.sleep(restart_delay)
                    break

    def _repair_and_retry() -> bool:
        if SelfRepair.attempt_repair():
            print("[launcher] Repair succeeded; restarting engine.")
            time.sleep(restart_delay)
            return True
        print("[launcher] Repair failed; sleeping then retrying.")
        time.sleep(restart_delay)
        return True

    while True:
        engine = None
        try:
            # Prompt once before model loading so backend auth is ready
            # immediately after startup.
            _maybe_prompt_for_api_key()
            _maybe_run_startup_training()

            # Construct engine in a background thread while the TUI shows progress.
            class EngineLoader:
                def __init__(self):
                    self.engine = None
                    self.progress = 0.0
                    self.message = "starting"
                    self.pass_list = {}
                    self._ready = False

                def report(self, frac, msg="", **kwargs):
                    try:
                        self.progress = float(frac)
                        self.message = str(msg)
                        # optional check updates
                        check = kwargs.get('check_name')
                        passed = kwargs.get('passed')
                        if check is not None:
                            self.pass_list[check] = bool(passed)
                    except Exception:
                        pass

                def set_engine(self, eng):
                    self.engine = eng
                    self._ready = True

                @property
                def ready(self):
                    return self._ready

            loader = EngineLoader()

            # Reduce noisy third-party logging during startup loader mode.
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

            def _make_engine():
                try:
                    # Suppress noisy third-party stdout/stderr during loader startup
                    # (sentence-transformers / huggingface warnings), which can
                    # corrupt the curses UI if written while TUI is active.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            eng = Engine(progress_cb=loader.report)
                    loader.set_engine(eng)
                except Exception:
                    # ensure loader shows failure
                    loader.report(1.0, "failed")

            import threading

            maker = threading.Thread(target=_make_engine, daemon=True)
            maker.start()

            # Pass the loader into the TUI; it will wait for the engine to become ready
            if _has_interactive_tty():
                try:
                    tui.run(loader)
                    engine = loader.engine
                    break
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"[launcher] TUI unavailable, falling back to REPL: {e}")
                    maker.join()
                    engine = loader.engine
                    if engine is None:
                        raise RuntimeError("engine failed to initialize")
                    _run_repl_loop(engine)
            else:
                # fallback: wait until engine is ready then set engine variable
                maker.join()
                engine = loader.engine
                if engine is None:
                    raise RuntimeError("engine failed to initialize")
                _run_repl_loop(engine)
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as e:
            print(f"[launcher] Fatal error while running engine: {e}")
            if _repair_and_retry():
                continue
        finally:
            if engine is not None:
                try:
                    del engine
                except Exception:
                    pass

    remove_pidfile()


if __name__ == "__main__":
    main()
