import os
import signal
import sys
import time

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

def _maybe_prompt_for_api_key() -> None:
    """(Legacy) prompt for an OpenAI key during launcher startup.

    Historically the launcher would ask for a key before constructing the
    engine.  This behaviour has been moved into the ASCII TUI so that the
    engine can be brought up regardless of whether a key is provided.  The
    helper remains only for backwards-compatibility and is exercised by the
    unit tests; `launcher.main` no longer calls it.
    """
    global _prompted_for_key
    if _prompted_for_key:
        return

    if sys.stdout.isatty() and not os.environ.get("OPENAI_API_KEY"):
        try:
            import getpass

            key = getpass.getpass(
                "Paste your OpenAI API key (leave empty to skip, not saved): "
            )
            # strip whitespace; empty means we explicitly do not want to
            # use a key for this session.  Remove any existing (empty)
            # environment variable to avoid re-prompting later.
            if key and key.strip():
                os.environ["OPENAI_API_KEY"] = key.strip()
            else:
                # explicit user choice to skip; clear any existing variable
                os.environ.pop("OPENAI_API_KEY", None)
                # let the user know the engine will continue without key
                print("[launcher] no API key provided, continuing without OpenAI access")
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
    while True:
        engine = None
        try:
            # key prompting is now handled in the TUI after the engine loads

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

            def _make_engine():
                try:
                    eng = Engine(progress_cb=loader.report)
                    loader.set_engine(eng)
                except Exception:
                    # ensure loader shows failure
                    loader.report(1.0, "failed")

            import threading

            maker = threading.Thread(target=_make_engine, daemon=True)
            maker.start()

            # Pass the loader into the TUI; it will wait for the engine to become ready
            if sys.stdout.isatty():
                try:
                    tui.run(loader)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"[launcher] TUI error: {e}")
                    # attempt repair
                    if SelfRepair.attempt_repair():
                        print("[launcher] Repair succeeded; restarting engine.")
                        time.sleep(restart_delay)
                        continue
                    else:
                        print("[launcher] Repair failed; sleeping then retrying.")
                        time.sleep(restart_delay)
                        continue
            else:
                # fallback: wait until engine is ready then set engine variable
                maker.join()
                engine = loader.engine
            # If stdout is attached to a terminal, run the curses-based TUI.
            if sys.stdout.isatty():
                try:
                    tui.run(engine)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"[launcher] TUI error: {e}")
                    # attempt repair
                    if SelfRepair.attempt_repair():
                        print("[launcher] Repair succeeded; restarting engine.")
                        time.sleep(restart_delay)
                        continue
                    else:
                        print("[launcher] Repair failed; sleeping then retrying.")
                        time.sleep(restart_delay)
                        continue
            else:
                # fallback to simple REPL when not a TTY
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
                        response = engine.respond(user_input)
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
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as e:
            print(f"[launcher] Fatal error while running engine: {e}")
            if SelfRepair.attempt_repair():
                print("[launcher] Repair succeeded; restarting engine.")
                time.sleep(restart_delay)
                continue
            else:
                print("[launcher] Repair failed; sleeping then retrying.")
                time.sleep(restart_delay)
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
