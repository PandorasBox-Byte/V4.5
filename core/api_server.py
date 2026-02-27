#!/usr/bin/env python3
"""Minimal HTTP API to interact with the Engine.

POST /chat  -> JSON {"text": "..."} returns {"reply": "..."}

This server is intentionally dependency-free and uses the stdlib.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import threading
from urllib.parse import urlparse

from core.engine_template import Engine


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "EvoAI/0.1"

    def _set_json(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/status":
            self._set_json(200)
            try:
                data = self.server.engine.status()
            except Exception:
                data = {"error": "could not read status"}
            self.wfile.write(json.dumps(data).encode())
        else:
            self._set_json(404)
            self.wfile.write(json.dumps({"error": "not found"}).encode())

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/chat":
            self._set_json(404)
            self.wfile.write(json.dumps({"error": "not found"}).encode())
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "invalid json"}).encode())
            return

        text = payload.get("text", "")
        if not isinstance(text, str):
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "text must be string"}).encode())
            return

        try:
            reply = self.server.engine.respond(text)
        except Exception as e:
            self._set_json(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        self._set_json(200)
        self.wfile.write(json.dumps({"reply": reply}).encode())


def run(addr="127.0.0.1", port=8000, quiet=False):
    srv = ThreadedHTTPServer((addr, port), ChatHandler)
    srv.engine = Engine()  # Use an internally-created Engine (legacy CLI entry)
    # Use double braces to include a literal JSON example in the f-string
    if not quiet:
        print(f"EvoAI API running on http://{addr}:{port} — POST /chat with JSON {{\"text\": ...}}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def run_server(engine, addr="127.0.0.1", port=8000, start_thread=True, quiet=False):
    """Run the API server using an existing `engine` instance.

    If `start_thread` is True, the HTTP server will be started in a daemon
    background thread and the Thread object is returned. Otherwise the call
    will block serving requests.
    """
    srv = ThreadedHTTPServer((addr, port), ChatHandler)
    srv.engine = engine
    if not quiet:
        print(f"EvoAI API running on http://{addr}:{port}")
    if start_thread:
        t = threading.Thread(target=srv.serve_forever, daemon=True, name="EvoAI-API")
        t.start()
        return t
    else:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        return None


if __name__ == "__main__":
    run()
