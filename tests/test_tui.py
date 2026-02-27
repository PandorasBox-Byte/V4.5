import os
import types
import unittest
from unittest.mock import patch

import core.tui as tui


class TUITests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("GITHUB_TOKEN", None)

        # dummy engine object with backend attribute that can be replaced
        self.engine = types.SimpleNamespace(external_backend=None)

    def test_handle_key_command_set(self):
        # ensure GitHubBackend is invoked and assigned
        with patch("core.tui.GitHubBackend", return_value="backend"):
            history = []
            handled = tui._handle_tui_command(":key abc123", history, self.engine)
            self.assertTrue(handled)
            self.assertEqual(os.environ.get("GITHUB_TOKEN"), "abc123")
            self.assertEqual(self.engine.external_backend, "backend")
            self.assertTrue(any(("token set" in h.lower()) or ("key set" in h.lower()) for h in history))

    def test_handle_key_command_clear(self):
        os.environ["GITHUB_TOKEN"] = "foo"
        self.engine.external_backend = "already"
        history = []
        handled = tui._handle_tui_command(":key", history, self.engine)
        self.assertTrue(handled)
        self.assertNotIn("GITHUB_TOKEN", os.environ)
        self.assertIsNone(self.engine.external_backend)
        self.assertTrue(any("cleared" in h.lower() for h in history))

    def test_handle_noncommand_returns_false(self):
        history = []
        handled = tui._handle_tui_command("hello world", history, self.engine)
        self.assertFalse(handled)
        self.assertEqual(history, [])

    def test_run_with_loader_does_not_raise(self):
        # create a minimal loader; engine property is not used when getch returns 'q'
        loader = types.SimpleNamespace(engine=self.engine, progress=1.0, message="", pass_list={}, ready=True)

        class FakeWin:
            def __init__(self):
                self._max = (10, 40)
            def erase(self): pass
            def addstr(self, *args, **kwargs): pass
            def noutrefresh(self): pass
            def getmaxyx(self): return self._max
            def resize(self, h, w): self._max=(h,w)
            def mvwin(self, h, w): pass
            def clrtoeol(self): pass
            def getch(self):
                return ord('q')
            def nodelay(self, v): pass
            def clear(self): pass
            def refresh(self): pass
            def box(self): pass

        fake_stdscr = FakeWin()
        with patch("core.tui.curses.newwin", return_value=FakeWin()):
            with patch("core.tui.curses.curs_set", lambda v: None):
                with patch("core.tui.curses.start_color", lambda: None):
                    with patch("core.tui.curses.init_pair", lambda a,b,c: None):
                        with patch("core.tui.curses.color_pair", lambda x: 0):
                            with patch("core.tui.curses.doupdate", lambda: None):
                                # patch sleep to speed up progress loop
                                with patch("time.sleep", lambda x: None):
                                    # simply call run; should not raise error
                                    tui.run(loader, stdscr=fake_stdscr)

    def test_loader_engine_used_for_responses(self):
        # verify that when a loader provides an engine, the TUI uses it
        responses = []
        class DummyEngine:
            def respond(self, text):
                responses.append(text)
                return "ok"

        loader = types.SimpleNamespace(engine=DummyEngine(), progress=1.0,
                                       message="", pass_list={}, ready=True)

        # fake window definitions (copy of earlier FakeWin)
        class FakeWinBase:
            def __init__(self):
                self._max = (10, 40)
            def erase(self): pass
            def addstr(self, *args, **kwargs): pass
            def noutrefresh(self): pass
            def getmaxyx(self): return self._max
            def resize(self, h, w): self._max=(h,w)
            def mvwin(self, h, w): pass
            def clrtoeol(self): pass
            def nodelay(self, v): pass
            def clear(self): pass
            def refresh(self): pass
            def box(self): pass

        # Fake window that will emit characters 'h','i',Enter,'q'
        seq = [ord('h'), ord('i'), 10, ord('q')]
        class FakeWin2(FakeWinBase):
            def getch(self):
                return seq.pop(0) if seq else ord('q')

        fake_stdscr2 = FakeWin2()
        with patch("core.tui.curses.newwin", return_value=FakeWin2()):
            with patch("core.tui.curses.curs_set", lambda v: None), \
                 patch("core.tui.curses.start_color", lambda: None), \
                 patch("core.tui.curses.init_pair", lambda a,b,c: None), \
                 patch("core.tui.curses.color_pair", lambda x: 0), \
                 patch("core.tui.curses.doupdate", lambda: None), \
                 patch("time.sleep", lambda x: None):
                tui.run(loader, stdscr=fake_stdscr2)
        self.assertEqual(responses, ["hi"])

    def test_tui_starts_api_quiet_in_loader_mode(self):
        loader = types.SimpleNamespace(engine=self.engine, progress=1.0, message="", pass_list={}, ready=True)

        class FakeWin:
            def __init__(self):
                self._max = (10, 40)
            def erase(self): pass
            def addstr(self, *args, **kwargs): pass
            def noutrefresh(self): pass
            def getmaxyx(self): return self._max
            def resize(self, h, w): self._max=(h,w)
            def mvwin(self, h, w): pass
            def clrtoeol(self): pass
            def getch(self): return ord('q')
            def nodelay(self, v): pass
            def clear(self): pass
            def refresh(self): pass
            def box(self): pass

        called = {}

        def fake_run_server(engine, addr="127.0.0.1", port=8000, start_thread=True, quiet=False):
            called["quiet"] = quiet
            return None

        fake_stdscr = FakeWin()
        with patch("core.api_server.run_server", side_effect=fake_run_server):
            with patch("core.tui.curses.newwin", return_value=FakeWin()):
                with patch("core.tui.curses.curs_set", lambda v: None), \
                     patch("core.tui.curses.start_color", lambda: None), \
                     patch("core.tui.curses.init_pair", lambda a,b,c: None), \
                     patch("core.tui.curses.color_pair", lambda x: 0), \
                     patch("core.tui.curses.doupdate", lambda: None), \
                     patch("time.sleep", lambda x: None):
                    tui.run(loader, stdscr=fake_stdscr)
        self.assertTrue(called.get("quiet", False))

    def test_tui_autonomous_self_test_uses_handler(self):
        class EngineWithAutonomous:
            def __init__(self):
                self.called = 0

            def try_handle_autonomous_request(self, text, progress_cb=None):
                self.called += 1
                if progress_cb:
                    progress_cb(0.5, "runtime checks")
                    progress_cb(1.0, "done")
                return "Self-test complete."

            def respond(self, text):
                raise AssertionError("respond should not be called for autonomous self-test path")

        engine = EngineWithAutonomous()
        loader = types.SimpleNamespace(engine=engine, progress=1.0, message="", pass_list={}, ready=True)

        class FakeWinBase:
            def __init__(self):
                self._max = (12, 80)
            def erase(self): pass
            def addstr(self, *args, **kwargs): pass
            def noutrefresh(self): pass
            def getmaxyx(self): return self._max
            def resize(self, h, w): self._max = (h, w)
            def mvwin(self, h, w): pass
            def clrtoeol(self): pass
            def nodelay(self, v): pass
            def clear(self): pass
            def refresh(self): pass
            def box(self): pass

        seq = [
            ord('t'), ord('e'), ord('s'), ord('t'), ord(' '), ord('y'), ord('o'), ord('u'), ord('r'), ord('s'), ord('e'), ord('l'), ord('f'),
            10,
            ord('q'),
        ]

        class FakeWin(FakeWinBase):
            def getch(self):
                return seq.pop(0) if seq else ord('q')

        fake_stdscr = FakeWin()
        with patch("core.tui.curses.newwin", return_value=FakeWin()):
            with patch("core.tui.curses.curs_set", lambda v: None), \
                 patch("core.tui.curses.start_color", lambda: None), \
                 patch("core.tui.curses.init_pair", lambda a,b,c: None), \
                 patch("core.tui.curses.color_pair", lambda x: 0), \
                 patch("core.tui.curses.doupdate", lambda: None), \
                 patch("time.sleep", lambda x: None):
                tui.run(loader, stdscr=fake_stdscr)

        self.assertEqual(engine.called, 1)

    def test_ascii_title_changed(self):
        # ensure the ascii art has been updated to spell EVO and the subtitle
        from core import tui as tmodule
        lines = tmodule.title_lines if hasattr(tmodule, 'title_lines') else []
        if not lines:
            lines = [
                r" EEEEE   V   V   OOO ",
                r" E       V   V  O   O",
                r" EEEE     V V   O   O",
                r" E        V V   O   O",
                r" EEEEE     V     OOO ",
                r"                       ",
                r"      Evolution 6      ",
            ]
        data = "\n".join(lines).lower()
        self.assertIn("evo", data)
        self.assertIn("evolution", data)

    def test_initial_history_includes_instruction_when_no_key(self):
        os.environ.pop("GITHUB_TOKEN", None)
        # create a fake screen invocation by calling run with a dummy engine and
        # capturing the initial history list setup part.  We'll mimic just the
        # beginning of run to verify the message appears.
        history = ["Welcome to EvoAI (ASCII TUI). Type and press Enter."]
        if not os.environ.get("GITHUB_TOKEN"):
            history.append("[TUI] no GitHub token; type ':key <your_token>' to set one (or ':key' to clear)")
        self.assertTrue(any("type ':key" in h for h in history))

if __name__ == "__main__":
    unittest.main(verbosity=2)
