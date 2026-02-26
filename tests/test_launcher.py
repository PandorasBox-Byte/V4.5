import io
import os
import sys
import unittest
from unittest.mock import patch

from core import launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        # ensure environment clean and reset prompt flag
        os.environ.pop("OPENAI_API_KEY", None)
        launcher._prompted_for_key = False

    def test_prompt_sets_key_when_provided(self):
        # first call should ask and set the key
        with patch("getpass.getpass", return_value="   mykey   ") as mock_getpass:
            fake_out = io.StringIO()
            fake_out.isatty = lambda: True
            with patch("sys.stdout", new=fake_out):
                launcher._maybe_prompt_for_api_key()
            mock_getpass.assert_called_once()
            self.assertEqual(os.environ.get("OPENAI_API_KEY"), "mykey")
        # second invocation should not call getpass again
        with patch("getpass.getpass", return_value="ignored") as mock_getpass:
            launcher._maybe_prompt_for_api_key()
            mock_getpass.assert_not_called()
            # key remains unchanged
            self.assertEqual(os.environ.get("OPENAI_API_KEY"), "mykey")

    def test_prompt_clears_key_when_empty(self):
        # simulate user pressing enter without providing key and capture stdout
        with patch("getpass.getpass", return_value="   ") as mock_getpass:
            fake_out = io.StringIO()
            # simulate being a TTY
            fake_out.isatty = lambda: True
            with patch("sys.stdout", new=fake_out):
                launcher._maybe_prompt_for_api_key()
            mock_getpass.assert_called_once()
            self.assertNotIn("OPENAI_API_KEY", os.environ)
            output = fake_out.getvalue()
            self.assertIn("no api key provided", output.lower())
        # subsequent calls should not re-prompt
        with patch("getpass.getpass", return_value="ignored") as mock_getpass:
            launcher._maybe_prompt_for_api_key()
            mock_getpass.assert_not_called()

    def test_main_does_not_prompt_early(self):
        # if _maybe_prompt_for_api_key were called during main(), this
        # patched version would raise AssertionError and the test would fail.
        called = False
        def fail_if_called():
            nonlocal called
            called = True
            raise AssertionError("should not prompt before loader")

        with patch("core.launcher._maybe_prompt_for_api_key", side_effect=fail_if_called):
            # patch Engine construction and the TUI to break out immediately
            with patch("core.launcher.Engine", new=lambda *args, **kwargs: types.SimpleNamespace(respond=lambda x: "", status=lambda: {})):
                with patch("core.launcher.tui.run", side_effect=KeyboardInterrupt):
                    # avoid writing pid files
                    with patch("core.launcher.write_pidfile"), patch("core.launcher.remove_pidfile"):
                        launcher.main()
        self.assertFalse(called, "_maybe_prompt_for_api_key was unexpectedly called")

    def test_no_prompt_if_key_already_set(self):
        os.environ["OPENAI_API_KEY"] = "foo"
        with patch("getpass.getpass", return_value="bar") as mock_getpass:
            launcher._maybe_prompt_for_api_key()
            mock_getpass.assert_not_called()
            self.assertEqual(os.environ.get("OPENAI_API_KEY"), "foo")


if __name__ == "__main__":
    unittest.main(verbosity=2)
