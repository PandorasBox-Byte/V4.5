"""Simple curses-based ASCII GUI for EvoAI.

Provides a text area for conversation history and an input box at the
bottom. Keeps minimal dependencies (stdlib curses) and is intentionally
lightweight to be a drop-in replacement for the plain REPL.
"""
from __future__ import annotations

import curses
import textwrap
import os
from typing import List

# allow tests to patch or replace the backend class easily
try:
    from core.openai_backend import OpenAIBackend
except Exception:  # pragma: no cover - openai optional
    OpenAIBackend = None


def _handle_tui_command(text: str, history: List[str], engine) -> bool:
    """Return True if *text* was a recognized command and was handled.

    Currently only ":key" is supported.  This allows the user to set or
    clear the OpenAI API key after the engine has already loaded.
    """
    if not text.lower().startswith(":key"):
        return False
    parts = text.split(None, 1)
    if len(parts) == 1 or not parts[1].strip():
        os.environ.pop("OPENAI_API_KEY", None)
        if hasattr(engine, "openai_backend"):
            engine.openai_backend = None
        history.append("[TUI] OpenAI API key cleared")
    else:
        val = parts[1].strip()
        os.environ["OPENAI_API_KEY"] = val
        # try to initialize backend if engine already created
        if OpenAIBackend is not None and hasattr(engine, "openai_backend"):
            try:
                engine.openai_backend = OpenAIBackend()
                history.append("[TUI] OpenAI API key set for session")
            except Exception as e:
                history.append(f"[TUI] failed to init OpenAI backend: {e}")
        else:
            history.append("[TUI] OpenAI API key set for session (backend unavailable)")
    return True


def _draw_history(win, lines: List[str], maxy: int, maxx: int, pad_top: int = 0):
    win.erase()
    y = pad_top
    for line in lines:
        for wrapped in textwrap.wrap(line, maxx - 2) or [""]:
            if y >= maxy - 1:
                break
            win.addstr(y, 1, wrapped)
            y += 1
        if y >= maxy - 1:
            break
    win.box()
    win.noutrefresh()


def _draw_input(win, prompt: str, buffer: str):
    win.erase()
    maxy, maxx = win.getmaxyx()
    win.addstr(0, 0, prompt)
    # truncate buffer if too long
    disp = buffer[-(maxx - len(prompt) - 1) :]
    try:
        win.addstr(0, len(prompt), disp)
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
        if not __import__("os").environ.get("OPENAI_API_KEY"):
            history.append("[TUI] no OpenAI key; type ':key <your_key>' to set one (or ':key' to clear)")
        buffer = ""
        prompt = "You: "
        input_history: List[str] = []
        history_index = 0

        # Optionally start the API server in a background thread unless disabled.
        try:
            import os

            if not os.environ.get("EVOAI_DISABLE_API"):
                try:
                    from core.api_server import run_server

                    api_thread = run_server(
                        engine,
                        addr=os.environ.get("EVOAI_API_ADDR", "127.0.0.1"),
                        port=int(os.environ.get("EVOAI_API_PORT", "8000")),
                        start_thread=True,
                    )
                except Exception:
                    api_thread = None
            else:
                api_thread = None
        except Exception:
            api_thread = None

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
            r"      Evoultion 6      ",
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
        except Exception:
            pass

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
                stdscr_inner.refresh()
            except curses.error:
                pass

        # run tests in background and update bar
        test_result = None
        test_progress = 0.0
        model_progress = 0.0

        if SelfRepair is not None:
            import threading

            def _run_tests():
                nonlocal test_result, test_progress
                ok, _out = SelfRepair.run_tests()
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
                test_progress = min(0.95, (elapsed / max(1.0, max_wait)) )
            elif test_progress < 1.0 and test_result is not None:
                test_progress = 1.0

            overall = min(1.0, model_progress * 0.7 + test_progress * 0.3)
            draw_bar(overall, True if test_result else None)

            # draw pass list under the bar
            try:
                pass_lines = []
                if is_loader and loader is not None:
                    # loader object has pass_list
                    pls = getattr(loader, 'pass_list', {}) or {}
                    for idx, (k, v) in enumerate(pls.items()):
                        status = '[OK]' if v else '[FAIL]'
                        pass_lines.append(f"{status} {k}")
                else:
                    # use local placeholder if available
                    pass_lines = []

                for i, pl in enumerate(pass_lines[:5]):
                    try:
                        stdscr_inner.addstr(bar_row + 3 + i, max(0, (maxx - len(pl)) // 2), pl)
                    except curses.error:
                        pass
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

        # Autonomous startup: appear to "wake up" and ask self-questions.
        try:
            import queue

            startup_queue = queue.Queue()

            def _cb(chunk, final):
                startup_queue.put((chunk, final))

            startup_prompt = (
                "You are EvoAI, a helpful assistant. Review your recent memory and the user's context and "
                "generate three concise, introspective questions you would ask the user to better understand "
                "their needs. Separate each question on its own line."
            )

            # At this point `engine` has already been set above when we
            # swapped out the loader.  Do not overwrite it again based on the
            # original argument – that was the source of the earlier bug where
            # `engine` became None after the startup phase.
            if engine and getattr(engine, "llm_model", None) and getattr(engine, "llm_tokenizer", None):
                thread = engine.generate_stream(startup_prompt, _cb, chunk_size=64)
                # consume queue and append to history while thread runs
                assembling = ""
                while True:
                    try:
                        chunk, final = startup_queue.get(timeout=0.1)
                        if chunk:
                            assembling += chunk
                            # update last EvoAI line or append
                            if not history or not history[-1].startswith("EvoAI:"):
                                history.append("EvoAI: " + assembling)
                            else:
                                history[-1] = "EvoAI: " + assembling
                            _draw_history(history_win, history, hist_h, maxx, pad_top=0)
                            curses.doupdate()
                        if final:
                            break
                    except queue.Empty:
                        if thread and not thread.is_alive():
                            break
                        continue
            else:
                # fallback canned introspective questions
                canned = [
                    "EvoAI: What is the most important thing you want help with today?",
                    "EvoAI: Are there recent files or notes I should consider?",
                    "EvoAI: How do you prefer responses — concise, detailed, or step-by-step?",
                ]
                for q in canned:
                    history.append(q)
                    _draw_history(history_win, history, hist_h, maxx, pad_top=0)
                    curses.doupdate()
                    time.sleep(0.25)
        except Exception:
            # don't let startup sequence crash the UI
            pass

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
            _draw_input(input_win, prompt, buffer)
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

    if stdscr is None:
        curses.wrapper(_main)
    else:
        _main(stdscr)


if __name__ == "__main__":
    # fallback quick-run that creates an Engine only when the module is executed
    from core.engine_template import Engine

    eng = Engine()
    run(eng)
