import http.server
import json
import os
import socketserver
import tempfile
import threading
import unittest
from functools import partial
from unittest.mock import patch

from core import auto_updater


def _serve_directory(directory):
    # start a simple HTTP server serving *directory*, return (thread, port, server)
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    srv = socketserver.TCPServer(("", 0), handler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return thread, port, srv


class UpdaterTests(unittest.TestCase):
    def setUp(self):
        # create a temporary "remote" repository
        self.remote = tempfile.TemporaryDirectory()
        self.remote_path = self.remote.name
        # prepare a fake file we intend to update (use project root rather than cwd)
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.target_file = os.path.join(root, "core", "update_test.txt")
        with open(self.target_file, "w") as f:
            f.write("original\n")

        # create updated version in remote dir
        os.makedirs(os.path.join(self.remote_path, "core"), exist_ok=True)
        with open(os.path.join(self.remote_path, "core", "update_test.txt"), "w") as f:
            f.write("updated\n")

        # manifest
        manifest = {
            "version": "v2",
            "files": [
                {"path": "core/update_test.txt", "url": ""},
            ],
        }
        # placeholder url; we will fill after starting server
        with open(os.path.join(self.remote_path, "manifest.json"), "w") as f:
            json.dump(manifest, f)

        self.server_thread, self.port, self.server_instance = _serve_directory(self.remote_path)
        # update manifest with real url
        manifest_url = f"http://127.0.0.1:{self.port}/manifest.json"
        manifest["files"][0]["url"] = f"http://127.0.0.1:{self.port}/core/update_test.txt"
        with open(os.path.join(self.remote_path, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        self.manifest_url = manifest_url

    def tearDown(self):
        self.remote.cleanup()
        # restore original file content
        with open(self.target_file, "w") as f:
            f.write("original\n")
        if hasattr(self, "server_instance"):
            self.server_instance.shutdown()
            self.server_instance.server_close()

    def test_run_update_flow_user_declines(self):
        # patch input to simulate user pressing n
        import builtins
        original_input = builtins.input
        builtins.input = lambda prompt="": "n"
        try:
            applied = auto_updater.run_update_flow(self.manifest_url)
            self.assertFalse(applied)
            with open(self.target_file) as f:
                data = f.read()
            self.assertEqual(data, "original\n")
        finally:
            import builtins
            builtins.input = original_input

    def test_run_update_flow_user_accepts(self):
        import builtins
        original_input = builtins.input
        builtins.input = lambda prompt="": "y"
        try:
            applied = auto_updater.run_update_flow(self.manifest_url)
            self.assertTrue(applied)
            with open(self.target_file) as f:
                data = f.read()
            self.assertEqual(data, "updated\n")
        finally:
            import builtins
            builtins.input = original_input

    def test_startup_git_update_normalizes_version_after_stash_pop(self):
        calls = []

        def fake_run_git(args, cwd=None, timeout=30):
            calls.append(list(args))
            if args[:3] == ["rev-parse", "--abbrev-ref", "HEAD"]:
                return 0, "main", ""
            if args[:3] == ["ls-remote", "--tags", "origin"]:
                return 0, "abc refs/tags/v5.1.2", ""
            if args[:3] == ["fetch", "--tags", "origin"]:
                return 0, "", ""
            if args[:2] == ["status", "--porcelain"]:
                return 0, " M setup.cfg\n", ""
            if args[:2] == ["stash", "push"]:
                return 0, "Saved working directory", ""
            if args[:2] == ["reset", "--hard"]:
                return 0, "", ""
            if args[:2] == ["stash", "pop"]:
                return 0, "Applied stash", ""
            if args[:2] == ["checkout", "v5.1.2"]:
                return 0, "", ""
            return 0, "", ""

        versions = iter(["5.1.1", "5.1.1", "5.1.1", "5.1.2"])

        def fake_local_version():
            try:
                return next(versions)
            except StopIteration:
                return "5.1.2"

        with patch("core.auto_updater.shutil.which", return_value="/usr/bin/git"):
            with patch("core.auto_updater._has_git_repo", return_value=True):
                with patch("core.auto_updater._run_git", side_effect=fake_run_git):
                    with patch("core.auto_updater.get_local_version", side_effect=fake_local_version):
                        result = auto_updater.run_startup_git_update(remote="origin")

        self.assertTrue(result.success)
        self.assertTrue(result.updated)
        self.assertTrue(result.needs_restart)
        self.assertIn(["checkout", "v5.1.2", "--", "version_tally.json", "setup.cfg"], calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
