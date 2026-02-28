import os
import signal
import sys
import time
import json
import shlex
import subprocess
import io
import contextlib
import warnings
import threading

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
from core.auto_updater import run_startup_git_update, get_local_version
from core import tui
from core.brain_monitor import launch_brain_monitor_async, _get_platform, _is_ssh_session

PIDFILE = os.path.join("data", "engine.pid")
_UPDATE_GUARD_ENV = "EVOAI_UPDATED_TARGET_VERSION"
_brain_monitor_process = None  # Track brain monitor subprocess


def _semver_tuple(value: str) -> tuple[int, int, int] | None:
    try:
        parts = value.strip().split(".")
        if len(parts) != 3:
            return None
        return int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:
        return None


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
    global _brain_monitor_process
    print("\nShutting down EvoAI (launcher cleanup)...")
    
    # Kill brain monitor if it's running
    if _brain_monitor_process is not None:
        try:
            _brain_monitor_process.terminate()
            _brain_monitor_process.wait(timeout=2)
        except Exception:
            try:
                _brain_monitor_process.kill()
            except Exception:
                pass
    
    remove_pidfile()
    sys.exit(0)


def _restart_launcher_process() -> None:
    remove_pidfile()
    os.execv(sys.executable, [sys.executable, *sys.argv])


# internal flag preventing repeated prompting
_prompted_for_key = False


def _token_env_file() -> str:
    return os.environ.get("EVOAI_TOKEN_ENV_FILE", os.path.join(os.path.expanduser("~"), ".evoai_env"))


def _load_saved_token() -> str | None:
    path = _token_env_file()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export GITHUB_TOKEN="):
                    _, value = line.split("=", 1)
                    parsed = shlex.split(value)
                    token = parsed[0] if parsed else value.strip().strip('"').strip("'")
                    token = token.strip()
                    return token or None
    except Exception:
        return None
    return None


def _persist_token(token: str) -> bool:
    path = _token_env_file()
    safe_token = token.strip()
    if not safe_token:
        return False
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("# EvoAI GitHub token\n")
            f.write(f"export GITHUB_TOKEN={safe_token!r}\n")
        os.chmod(path, 0o600)
        return True
    except Exception:
        return False


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

    env_token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
    if env_token:
        os.environ["GITHUB_TOKEN"] = env_token
        os.environ.pop("GH_TOKEN", None)
        _prompted_for_key = True
        return

    stdin_tty = getattr(sys.stdin, "isatty", lambda: False)()
    stdout_tty = getattr(sys.stdout, "isatty", lambda: False)()
    if not (stdin_tty and stdout_tty):
        _prompted_for_key = True
        return

    if stdout_tty:
        try:
            import getpass

            saved_token = _load_saved_token()
            base_token = saved_token or ""

            if base_token:
                try:
                    choice = input("Use saved GitHub token? [Y=use / C=change / S=skip]: ").strip().lower()
                except EOFError:
                    choice = "y"
                if choice in ("", "y", "yes"):
                    os.environ["GITHUB_TOKEN"] = base_token
                    os.environ.pop("GH_TOKEN", None)
                elif choice in ("c", "change", "n", "no"):
                    try:
                        key = getpass.getpass("Paste new GitHub token (leave empty to skip): ")
                    except EOFError:
                        key = ""
                    if key and key.strip():
                        token = key.strip()
                        os.environ["GITHUB_TOKEN"] = token
                        os.environ.pop("GH_TOKEN", None)
                        if _persist_token(token):
                            print(f"[launcher] token saved to {_token_env_file()}")
                    else:
                        os.environ.pop("GITHUB_TOKEN", None)
                        os.environ.pop("GH_TOKEN", None)
                        print("[launcher] no token provided, continuing without GitHub backend access")
                else:
                    os.environ.pop("GITHUB_TOKEN", None)
                    os.environ.pop("GH_TOKEN", None)
                    print("[launcher] continuing without GitHub token")
            else:
                try:
                    key = getpass.getpass("Paste your GitHub token (leave empty to skip): ")
                except EOFError:
                    key = ""
                if key and key.strip():
                    token = key.strip()
                    os.environ["GITHUB_TOKEN"] = token
                    os.environ.pop("GH_TOKEN", None)
                    if _persist_token(token):
                        print(f"[launcher] token saved to {_token_env_file()}")
                else:
                    os.environ.pop("GITHUB_TOKEN", None)
                    os.environ.pop("GH_TOKEN", None)
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
        platform = _get_platform()
        session_type = "SSH session" if _is_ssh_session() else "local session"
        print(f"Engine started on {platform} ({session_type}) — using REPL.")
        while True:
            try:
                user_input = input("You: ")
            except EOFError:
                raise SystemExit
            except OSError as e:
                msg = str(e).lower()
                if "pytest: reading from stdin while output is captured" in msg or os.environ.get("PYTEST_CURRENT_TEST"):
                    raise SystemExit
                print(f"[launcher] REPL input unavailable: {e}")
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

            # Launch brain monitor in separate terminal window (if enabled)
            if os.environ.get("EVOAI_ENABLE_BRAIN_MONITOR", "1").lower() in ("1", "true", "yes"):
                try:
                    global _brain_monitor_process
                    _brain_monitor_process = launch_brain_monitor_async(os.getcwd())
                    if _brain_monitor_process is None:
                        # launch_brain_monitor_async already printed reason
                        pass
                except Exception as e:
                    print(f"[launcher] Brain monitor launch error: {e}")

            # Construct engine in a background thread while the TUI shows progress.
            class EngineLoader:
                def __init__(self):
                    self.engine = None
                    self.progress = 0.0
                    self.message = "starting"
                    self.phase = "startup"
                    self.pass_list = {}
                    self.update_done = False
                    self.update_success = False
                    self.update_error = ""
                    self.restart_requested = False
                    self._ready = False

                def report(self, frac, msg="", **kwargs):
                    try:
                        self.progress = float(frac)
                        self.message = str(msg)
                        phase = kwargs.get("phase")
                        if phase:
                            self.phase = str(phase)
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

            def _run_update_phase():
                enabled = os.environ.get("EVOAI_ENABLE_STARTUP_GIT_UPDATE", "1").lower() in ("1", "true", "yes")
                if not enabled:
                    loader.update_done = True
                    return

                guard_target = (os.environ.get(_UPDATE_GUARD_ENV) or "").strip()
                if guard_target:
                    local_ver = (get_local_version() or "").strip().lstrip("v")
                    local_tuple = _semver_tuple(local_ver)
                    guard_tuple = _semver_tuple(guard_target.lstrip("v"))
                    if local_tuple is not None and guard_tuple is not None and local_tuple >= guard_tuple:
                        os.environ.pop(_UPDATE_GUARD_ENV, None)
                        loader.update_success = True
                        loader.update_done = True
                        loader.report(1.0, f"Update already applied: v{local_ver}", phase="update")
                        return

                remote = os.environ.get("EVOAI_GIT_REMOTE", "origin")

                def _update_progress(frac: float, msg: str):
                    loader.report(frac, msg, phase="update")

                result = run_startup_git_update(progress_cb=_update_progress, remote=remote)
                loader.update_success = bool(result.success)
                loader.update_error = result.failed_reason or ""
                loader.update_done = True

                if result.updated and result.success and result.needs_restart:
                    loader.restart_requested = True
                    if result.remote_version:
                        os.environ[_UPDATE_GUARD_ENV] = str(result.remote_version)
                    loader.report(1.0, f"Update successful: v{result.remote_version}", phase="update")
                elif not result.success:
                    os.environ.pop(_UPDATE_GUARD_ENV, None)
                    loader.report(1.0, f"Update failed: {loader.update_error}", phase="update")
                else:
                    os.environ.pop(_UPDATE_GUARD_ENV, None)
                    loader.report(1.0, "No update required", phase="update")

            update_thread = threading.Thread(target=_run_update_phase, daemon=True)
            update_thread.start()

            # Reduce noisy third-party logging during startup loader mode.
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

            def _make_engine():
                try:
                    loader.report(0.0, "Loading engine", phase="startup")
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

            maker = threading.Thread(target=_make_engine, daemon=True)
            maker.start()

            # Pass the loader into the TUI; it will wait for the engine to become ready
            if _has_interactive_tty():
                try:
                    tui_action = tui.run(loader)
                    if tui_action == "restart" or loader.restart_requested:
                        _restart_launcher_process()
                    engine = loader.engine
                    break
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"[launcher] TUI unavailable, falling back to REPL: {e}")
                    maker.join()
                    if loader.restart_requested:
                        _restart_launcher_process()
                    engine = loader.engine
                    if engine is None:
                        raise RuntimeError("engine failed to initialize")
                    _run_repl_loop(engine)
            else:
                # fallback: wait until engine is ready then set engine variable
                maker.join()
                while not loader.update_done:
                    time.sleep(0.05)
                if loader.restart_requested:
                    _restart_launcher_process()
                engine = loader.engine
                if engine is None:
                    raise RuntimeError("engine failed to initialize")
                _run_repl_loop(engine)
                break
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

    # Cleanup brain monitor on exit
    if _brain_monitor_process is not None:
        try:
            _brain_monitor_process.terminate()
            _brain_monitor_process.wait(timeout=2)
        except Exception:
            try:
                _brain_monitor_process.kill()
            except Exception:
                pass
    
    remove_pidfile()


if __name__ == "__main__":
    main()
