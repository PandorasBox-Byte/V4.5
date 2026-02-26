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


def main():
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    engine = Engine()
    write_pidfile()

    try:
        while True:
            try:
                user_input = input("You: ")
            except EOFError:
                break
            response = engine.respond(user_input)
            print("EvoAI:", response)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        remove_pidfile()


if __name__ == "__main__":
    main()
