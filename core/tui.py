"""Simple curses-based ASCII GUI for EvoAI.

Provides a text area for conversation history and an input box at the
bottom. Keeps minimal dependencies (stdlib curses) and is intentionally
lightweight to be a drop-in replacement for the plain REPL.
"""
from __future__ import annotations

import curses
import textwrap
import os
import io
import json
import contextlib
import warnings
from typing import List
from pathlib import Path

# allow tests to patch or replace the backend class easily
try:
    from core.github_backend import GitHubBackend
except Exception:  # pragma: no cover - backend optional
    GitHubBackend = None


def _read_version_label() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    tally_path = repo_root / "version_tally.json"
    setup_cfg_path = repo_root / "setup.cfg"

    try:
        if tally_path.exists():
            data = json.loads(tally_path.read_text(encoding="utf-8"))
            current = str(data.get("current_version", "")).strip()
            if current:
                major = current.split(".", 1)[0]
                return f"{current} (V{major})"
    except Exception:
        pass

    try:
        if setup_cfg_path.exists():
            for raw in setup_cfg_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if line.startswith("version") and "=" in line:
                    current = line.split("=", 1)[1].strip()
                    if current:
                        major = current.split(".", 1)[0]
                        return f"{current} (V{major})"
    except Exception:
        pass

    return "unknown"


VERSION_LABEL = _read_version_label()


def _handle_tui_command(text: str, history: List[str], engine) -> bool:
    """Return True if *text* was a recognized command and was handled.

    Currently only ":key" is supported.  This allows the user to set or
    clear the GitHub token after the engine has already loaded.
    """
    if not text.lower().startswith(":key"):
        return False
    parts = text.split(None, 1)
    if len(parts) == 1 or not parts[1].strip():
        os.environ.pop("GITHUB_TOKEN", None)
        os.environ.pop("GH_TOKEN", None)
        if hasattr(engine, "external_backend"):
            engine.external_backend = None
        history.append("[TUI] GitHub token cleared")
    else:
        val = parts[1].strip()
        os.environ["GITHUB_TOKEN"] = val
        os.environ["EVOAI_BACKEND_PROVIDER"] = "github"
        # try to initialize backend if engine already created
        if GitHubBackend is not None and hasattr(engine, "external_backend"):
            try:
                engine.external_backend = GitHubBackend()
                history.append("[TUI] GitHub token set for session")
            except Exception as e:
                history.append(f"[TUI] failed to init GitHub backend: {e}")
        else:
            history.append("[TUI] GitHub token set for session (backend unavailable)")
    return True


def _draw_history(win, lines: List[str], maxy: int, maxx: int, pad_top: int = 0):
    win.erase()
    y = pad_top
    for line in lines:
        is_selftest_bar = isinstance(line, str) and line.startswith("[SelfTest][bar]")
        display_line = line
        if is_selftest_bar:
            display_line = line.replace("[SelfTest][bar]", "", 1).strip()

        for wrapped in textwrap.wrap(display_line, maxx - 2) or [""]:
            if y >= maxy - 1:
                break
            try:
                if is_selftest_bar:
                    win.addstr(y, 1, wrapped[: max(0, maxx - 2)], curses.color_pair(3))
                else:
                    win.addstr(y, 1, wrapped[: max(0, maxx - 2)])
            except curses.error:
                pass
            y += 1
        if y >= maxy - 1:
            break
    win.box()
    win.noutrefresh()


def _draw_input(win, prompt: str, buffer: str, version_label: str = ""):
    win.erase()
    maxy, maxx = win.getmaxyx()
    try:
        win.addstr(0, 0, prompt[:maxx])
    except curses.error:
        pass
    # truncate buffer if too long
    disp = buffer[-(maxx - len(prompt) - 1) :]
    try:
        win.addstr(0, len(prompt), disp)
    except curses.error:
        pass
    if version_label:
        try:
            win.addstr(maxy - 1, 0, version_label[:maxx])
        except curses.error:
            pass
    win.clrtoeol()
    win.noutrefresh()


def run(engine_or_loader, stdscr=None):
    def _main(stdscr_inner):
        curses.curs_set(1)
        stdscr_inner.nodelay(False)
        stdscr_inner.clear()

        # determine whether we were passed a loader or an actual engine
        is_loader = not hasattr(engine_or_loader, "respond")
        loader = engine_or_loader if is_loader else None
        engine = None if is_loader else engine_or_loader

        maxy, maxx = stdscr_inner.getmaxyx()
        hist_h = maxy - 3

        history_win = curses.newwin(hist_h, maxx, 0, 0)
        input_win = curses.newwin(3, maxx, hist_h, 0)

        history: List[str] = ["Welcome to EvoAI (ASCII TUI). Type and press Enter."]
        # if no API key available, tell the user how to set one later
        if not (__import__("os").environ.get("GITHUB_TOKEN") or __import__("os").environ.get("GH_TOKEN")):
            history.append("[TUI] no GitHub token; type ':key <your_token>' to set one (or ':key' to clear)")
        buffer = ""
        prompt = "You: "
        input_history: List[str] = []
        history_index = 0

        # Optionally start the API server in a background thread unless disabled.
        if not os.environ.get("EVOAI_DISABLE_API"):
            try:
                from core.api_server import run_server

                api_quiet = is_loader or os.environ.get("EVOAI_TUI_ACTIVE", "0") == "1"
                run_server(
                    engine,
                    addr=os.environ.get("EVOAI_API_ADDR", "127.0.0.1"),
                    port=int(os.environ.get("EVOAI_API_PORT", "8000")),
                    start_thread=True,
                    quiet=api_quiet,
                )
            except Exception:
                pass

        # Loading screen: display big ASCII title and a progress bar while
        # running a short startup test and (optionally) waiting for the
        # engine to finish loading. Progress is computed as a weighted sum
        # of model loading (70%) and self-tests (30%). Progress bar turns
        # green when tests pass.
        try:
            from core.self_repair import SelfRepair
        except Exception:
            SelfRepair = None

        maxy, maxx = stdscr_inner.getmaxyx()
        # draw title centered
        title_lines = [
            r" EEEEE   V   V   OOO ",
            r" E       V   V  O   O",
            r" EEEE     V V   O   O",
            r" E        V V   O   O",
            r" EEEEE     V     OOO ",
            r"                       ",  # blank line for spacing
            r"      Evolution 6      ",
        ]

        stdscr_inner.erase()
        start_row = max(0, (maxy - len(title_lines) - 4) // 2)
        for idx, line in enumerate(title_lines):
            try:
                stdscr_inner.addstr(start_row + idx, max(0, (maxx - len(line)) // 2), line)
            except curses.error:
                pass

        bar_w = min(60, maxx - 10)
        bar_row = start_row + len(title_lines) + 1
        stdscr_inner.addstr(bar_row + 1, max(0, (maxx - 20) // 2), "Starting up...")
        stdscr_inner.refresh()

        # init color pair for green/normal
        try:
            curses.start_color()
            curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_GREEN)
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_RED)
            curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
        except Exception:
            pass

        test_message = "Starting up..."

        def draw_bar(frac: float, ok: bool | None = None):
            filled = int(frac * bar_w)
            empty = bar_w - filled
            x = max(0, (maxx - bar_w) // 2)
            try:
                stdscr_inner.addstr(bar_row, x, "[")
                if ok is True:
                    stdscr_inner.addstr(bar_row, x + 1, "#" * filled, curses.color_pair(1))
                elif ok is False:
                    stdscr_inner.addstr(bar_row, x + 1, "#" * filled, curses.color_pair(2))
                else:
                    stdscr_inner.addstr(bar_row, x + 1, "#" * filled)
                stdscr_inner.addstr(bar_row, x + 1 + filled, "-" * empty)
                stdscr_inner.addstr(bar_row, x + 1 + bar_w, "]")
                msg = (test_message or "").strip()
                stdscr_inner.addstr(bar_row + 1, x, " " * bar_w)
                if msg:
                    stdscr_inner.addstr(bar_row + 1, max(0, (maxx - len(msg)) // 2), msg)
                stdscr_inner.refresh()
            except curses.error:
                pass

        def draw_pass_grid(entries):
            rows_to_clear = 6
            grid_y = bar_row + 3
            left = max(0, (maxx - bar_w) // 2)

            for row in range(rows_to_clear):
                try:
                    stdscr_inner.addstr(grid_y + row, left, " " * bar_w)
                except curses.error:
                    pass

            if not entries:
                return

            cols = 2 if bar_w >= 44 else 1
            gap = 2
            cell_w = max(12, (bar_w - (gap * (cols - 1))) // cols)
            max_cells = rows_to_clear * cols

            for idx, (name, passed) in enumerate(entries[:max_cells]):
                row = idx // cols
                col = idx % cols
                x = left + col * (cell_w + gap)
                status = "[OK]" if passed else "[FAIL]"
                text = f"{status} {name}"
                try:
                    stdscr_inner.addstr(grid_y + row, x, text[:cell_w])
                except curses.error:
                    pass

        # run tests in background and update bar
        test_result = None
        test_progress = 0.0
        model_progress = 0.0
        startup_pass_list = {}

        if SelfRepair is not None:
            import threading

            def _run_tests():
                nonlocal test_result, test_progress, test_message

                def _progress(frac, msg="", check_name=None, passed=None):
                    nonlocal test_progress, test_message
                    try:
                        test_progress = max(test_progress, float(frac))
                    except Exception:
                        pass
                    if msg:
                        test_message = str(msg)
                    if check_name is not None and passed is not None:
                        startup_pass_list[str(check_name)] = bool(passed)
                        if is_loader and loader is not None:
                            try:
                                loader.pass_list[str(check_name)] = bool(passed)
                            except Exception:
                                pass

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        ok, _out = SelfRepair.run_tests(
                            progress_cb=_progress,
                            mode="startup",
                            include_pytest=False,
                        )
                test_result = ok
                test_progress = 1.0

            t = threading.Thread(target=_run_tests, daemon=True)
            t.start()
        else:
            t = None

        # If the argument is an EngineLoader (created by launcher), it
        # provides `progress` (0..1) and `message`, and will have `.engine`
        # when ready.  We already determined `is_loader`, `loader`, and `engine`
        # above.

        import time
        start_ts = time.time()
        max_wait = int((os.environ.get("EVOAI_STARTUP_TIMEOUT", "30")))

        # progress loop: combine model (70%) and tests (30%) into overall bar
        while True:
            # model progress
            if is_loader and loader is not None:
                try:
                    model_progress = float(getattr(loader, "progress", 0.0))
                except Exception:
                    model_progress = 0.0
            else:
                # if not a loader, model already loaded
                model_progress = 1.0

            # animate test progress while tests are running
            if t is not None and t.is_alive():
                # slowly ramp test progress up to 0.95 until done
                elapsed = time.time() - start_ts
                # map elapsed to a soft progress for tests (bounded)
                test_progress = max(test_progress, min(0.95, (elapsed / max(1.0, max_wait))))
            elif test_progress < 1.0 and test_result is not None:
                test_progress = 1.0

            overall = min(1.0, model_progress * 0.7 + test_progress * 0.3)
            draw_bar(overall, True if test_result else None)

            # draw pass list under the bar
            try:
                pass_items = []
                if is_loader and loader is not None:
                    # loader object has pass_list
                    pls = getattr(loader, 'pass_list', {}) or {}
                    for k, v in pls.items():
                        pass_items.append((str(k), bool(v)))
                else:
                    for k, v in startup_pass_list.items():
                        pass_items.append((str(k), bool(v)))

                draw_pass_grid(pass_items)
                stdscr_inner.refresh()
            except Exception:
                pass

            # if loader and engine ready, break; if no loader and tests done, break
            if is_loader and loader is not None:
                if getattr(loader, "ready", False) or (time.time() - start_ts) > max_wait:
                    # set final bar to indicate test result
                    draw_bar(1.0, True if test_result else False)
                    time.sleep(0.35)
                    break
            else:
                if (t is None) or (test_result is not None) or (time.time() - start_ts) > max_wait:
                    draw_bar(1.0, True if test_result else False)
                    time.sleep(0.35)
                    break

            time.sleep(0.08)

        # If we were passed a loader, swap it out for the real Engine instance
        if is_loader and loader is not None:
            try:
                real = getattr(loader, "engine", None)
                if real is not None:
                    engine = real
                else:
                    # loader wasn't ready; proceed without an engine to avoid attribute errors
                    engine = None
            except Exception:
                engine = None
        else:
            # no tests available; show full bar
            draw_bar(1.0, True)
            time.sleep(0.35)

        while True:
            maxy, maxx = stdscr_inner.getmaxyx()
            hist_h = maxy - 3
            # resize windows if terminal changed
            try:
                history_win.resize(hist_h, maxx)
                input_win.resize(3, maxx)
                input_win.mvwin(hist_h, 0)
            except Exception:
                pass

            _draw_history(history_win, history, hist_h, maxx, pad_top=0)
            _draw_input(input_win, prompt, buffer, VERSION_LABEL)
            curses.doupdate()

            ch = input_win.getch()
            if ch in (curses.KEY_ENTER, 10, 13):
                text = buffer.strip()
                buffer = ""
                if not text:
                    continue
                history.append(f"You: {text}")
                # record into input history
                input_history.append(text)
                history_index = len(input_history)

                # intercept special commands (':key' modifies the API key)
                if _handle_tui_command(text, history, engine):
                    # command consumed, skip engine respond
                    continue

                try:
                    autonomous_handler = getattr(engine, "try_handle_autonomous_request", None)
                except Exception:
                    autonomous_handler = None

                if callable(autonomous_handler):
                    bar_idx = None

                    def _selftest_progress(frac, msg="", check_name=None, passed=None):
                        nonlocal bar_idx
                        try:
                            pct = max(0, min(100, int(float(frac) * 100)))
                        except Exception:
                            pct = 0
                        total = 20
                        filled = max(0, min(total, int((pct / 100.0) * total)))
                        bar = ("*" * filled) + ("-" * (total - filled))
                        tail = str(msg or "").strip()
                        line = f"[SelfTest][bar] [{bar}] {pct}% {tail}".rstrip()
                        if bar_idx is None:
                            history.append(line)
                            bar_idx = len(history) - 1
                        else:
                            history[bar_idx] = line
                        _draw_history(history_win, history, hist_h, maxx, pad_top=0)
                        _draw_input(input_win, prompt, buffer, VERSION_LABEL)
                        curses.doupdate()

                    try:
                        autonomous_reply = autonomous_handler(text, _selftest_progress)
                    except Exception:
                        autonomous_reply = None

                    if autonomous_reply is not None:
                        history.append(f"EvoAI: {autonomous_reply}")
                        if len(history) > 1000:
                            history = history[-1000:]
                        continue

                try:
                    reply = engine.respond(text)
                except Exception as e:
                    reply = f"(error) {e}"
                history.append(f"EvoAI: {reply}")
                # keep history reasonably bounded
                if len(history) > 1000:
                    history = history[-1000:]
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                buffer = buffer[:-1]
            elif ch in (curses.KEY_UP,):
                # navigate up in input history
                if input_history:
                    history_index = max(0, history_index - 1)
                    buffer = input_history[history_index]
            elif ch in (curses.KEY_DOWN,):
                if input_history:
                    history_index = min(len(input_history), history_index + 1)
                    if history_index == len(input_history):
                        buffer = ""
                    else:
                        buffer = input_history[history_index]
            elif ch in (curses.ascii.ctrl(ord('l')) if hasattr(curses, 'ascii') else 12,):
                # Ctrl-L clears the history
                history = []
                buffer = ""
            elif ch == 3 or ch in (ord('q'), ord('Q')):  # Ctrl-C or 'q' to quit
                break
            elif ch == curses.KEY_RESIZE:
                continue
            elif ch == -1:
                continue
            else:
                try:
                    buffer += chr(ch)
                except Exception:
                    pass

    prev_tui = os.environ.get("EVOAI_TUI_ACTIVE")
    os.environ["EVOAI_TUI_ACTIVE"] = "1"
    try:
        if stdscr is None:
            curses.wrapper(_main)
        else:
            _main(stdscr)
    finally:
        if prev_tui is None:
            os.environ.pop("EVOAI_TUI_ACTIVE", None)
        else:
            os.environ["EVOAI_TUI_ACTIVE"] = prev_tui


if __name__ == "__main__":
    # fallback quick-run that creates an Engine only when the module is executed
    from core.engine_template import Engine

    eng = Engine()
    run(eng)
