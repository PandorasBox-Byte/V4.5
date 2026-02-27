import io
import os
import sys
import types
import unittest
import tempfile
import json
from unittest.mock import patch

from core import launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        # ensure environment clean and reset prompt flag
        os.environ.pop("GITHUB_TOKEN", None)
        os.environ.pop("GH_TOKEN", None)
        launcher._prompted_for_key = False
        self._tmp = tempfile.TemporaryDirectory()
        self._token_file = os.path.join(self._tmp.name, ".evoai_env")
        os.environ["EVOAI_TOKEN_ENV_FILE"] = self._token_file

    def tearDown(self):
        self._tmp.cleanup()
        os.environ.pop("EVOAI_TOKEN_ENV_FILE", None)

    def test_startup_training_runs_at_threshold(self):
        with tempfile.TemporaryDirectory() as td:
            meta = os.path.join(td, "meta.json")
            with open(meta, "w", encoding="utf-8") as f:
                json.dump({"new_samples_since_train": 3}, f)

            old_meta = os.environ.get("EVOAI_TRAINING_META_PATH")
            old_thr = os.environ.get("EVOAI_STARTUP_TRAIN_THRESHOLD")
            old_auto = os.environ.get("EVOAI_STARTUP_AUTO_TRAIN")
            try:
                os.environ["EVOAI_TRAINING_META_PATH"] = meta
                os.environ["EVOAI_STARTUP_TRAIN_THRESHOLD"] = "3"
                os.environ["EVOAI_STARTUP_AUTO_TRAIN"] = "1"

                with patch("core.launcher.subprocess.run", return_value=types.SimpleNamespace(returncode=0)) as mock_run:
                    launcher._maybe_run_startup_training()
                mock_run.assert_called_once()

                with open(meta, "r", encoding="utf-8") as f:
                    updated = json.load(f)
                self.assertEqual(updated.get("new_samples_since_train"), 0)
                self.assertTrue(updated.get("last_train_ok"))
            finally:
                if old_meta is None:
                    os.environ.pop("EVOAI_TRAINING_META_PATH", None)
                else:
                    os.environ["EVOAI_TRAINING_META_PATH"] = old_meta
                if old_thr is None:
                    os.environ.pop("EVOAI_STARTUP_TRAIN_THRESHOLD", None)
                else:
                    os.environ["EVOAI_STARTUP_TRAIN_THRESHOLD"] = old_thr
                if old_auto is None:
                    os.environ.pop("EVOAI_STARTUP_AUTO_TRAIN", None)
                else:
                    os.environ["EVOAI_STARTUP_AUTO_TRAIN"] = old_auto

    def test_prompt_sets_key_when_provided(self):
        # first call should ask, set, and persist the key
        with patch("getpass.getpass", return_value="   mykey   ") as mock_getpass:
            fake_out = io.StringIO()
            fake_out.isatty = lambda: True
            with patch("sys.stdout", new=fake_out):
                launcher._maybe_prompt_for_api_key()
            mock_getpass.assert_called_once()
            self.assertEqual(os.environ.get("GITHUB_TOKEN"), "mykey")
            self.assertTrue(os.path.exists(self._token_file))
            with open(self._token_file, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("GITHUB_TOKEN", content)
        # second invocation should not call getpass again
        with patch("getpass.getpass", return_value="ignored") as mock_getpass:
            launcher._maybe_prompt_for_api_key()
            mock_getpass.assert_not_called()
            # key remains unchanged
            self.assertEqual(os.environ.get("GITHUB_TOKEN"), "mykey")

    def test_prompt_clears_key_when_empty(self):
        # simulate user pressing enter without providing key and capture stdout
        with patch("getpass.getpass", return_value="   ") as mock_getpass:
            fake_out = io.StringIO()
            # simulate being a TTY
            fake_out.isatty = lambda: True
            with patch("sys.stdout", new=fake_out):
                launcher._maybe_prompt_for_api_key()
            mock_getpass.assert_called_once()
            self.assertNotIn("GITHUB_TOKEN", os.environ)
            output = fake_out.getvalue()
            self.assertIn("no token provided", output.lower())
        # subsequent calls should not re-prompt
        with patch("getpass.getpass", return_value="ignored") as mock_getpass:
            launcher._maybe_prompt_for_api_key()
            mock_getpass.assert_not_called()

    def test_main_prompts_before_engine_load(self):
        called = False

        def mark_called():
            nonlocal called
            called = True

        def engine_ctor(*args, **kwargs):
            self.assertTrue(called, "expected key prompt before Engine construction")
            return types.SimpleNamespace(respond=lambda x: "", status=lambda: {})

        old_force_tui = os.environ.get("EVOAI_FORCE_TUI")
        try:
            os.environ["EVOAI_FORCE_TUI"] = "0"
            with patch("core.launcher._maybe_prompt_for_api_key", side_effect=mark_called):
                with patch("core.launcher.Engine", new=engine_ctor):
                    with patch("core.launcher.tui.run", side_effect=KeyboardInterrupt):
                        with patch("core.launcher.write_pidfile"), patch("core.launcher.remove_pidfile"):
                            launcher.main()
        finally:
            if old_force_tui is None:
                os.environ.pop("EVOAI_FORCE_TUI", None)
            else:
                os.environ["EVOAI_FORCE_TUI"] = old_force_tui
        self.assertTrue(called, "expected _maybe_prompt_for_api_key to be called")

    def test_no_prompt_if_key_already_set(self):
        os.environ["GITHUB_TOKEN"] = "foo"
        fake_out = io.StringIO()
        fake_out.isatty = lambda: True
        with patch("builtins.input", return_value="y") as mock_input:
            with patch("getpass.getpass", return_value="bar") as mock_getpass:
                with patch("sys.stdout", new=fake_out):
                    launcher._maybe_prompt_for_api_key()
        mock_input.assert_called_once()
        mock_getpass.assert_not_called()
        self.assertEqual(os.environ.get("GITHUB_TOKEN"), "foo")

    def test_prompt_with_saved_token_change_updates_file(self):
        with open(self._token_file, "w", encoding="utf-8") as f:
            f.write("export GITHUB_TOKEN='oldtoken'\n")

        fake_out = io.StringIO()
        fake_out.isatty = lambda: True
        with patch("builtins.input", return_value="c") as mock_input:
            with patch("getpass.getpass", return_value="newtoken") as mock_getpass:
                with patch("sys.stdout", new=fake_out):
                    launcher._maybe_prompt_for_api_key()

        mock_input.assert_called_once()
        mock_getpass.assert_called_once()
        self.assertEqual(os.environ.get("GITHUB_TOKEN"), "newtoken")
        with open(self._token_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("newtoken", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
