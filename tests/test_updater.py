import http.server
import json
import os
import socketserver
import tempfile
import threading
import unittest

from core import auto_updater


def _serve_directory(directory):
    # start a simple HTTP server serving *directory*, return (thread, port, server)
    handler = http.server.SimpleHTTPRequestHandler
    os.chdir(directory)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
