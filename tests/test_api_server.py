import json
import threading
import unittest
from urllib.request import Request, urlopen

from core.api_server import ThreadedHTTPServer, ChatHandler


class _StubEngine:
    def __init__(self):
        self._paused = False
        self._budget_max = 5
        self._budget_remaining = 5
        self._events = []

    def status(self):
        return {"ready": "yes", "autonomy_paused": str(self._paused).lower()}

    def governance_status(self):
        return {
            "autonomy_paused": self._paused,
            "autonomy_budget_max": self._budget_max,
            "autonomy_budget_remaining": self._budget_remaining,
            "audit_events": len(self._events),
        }

    def update_governance(self, payload):
        if "autonomy_paused" in payload:
            self._paused = bool(payload["autonomy_paused"])
        if "autonomy_budget_remaining" in payload:
            self._budget_remaining = int(payload["autonomy_budget_remaining"])
        return self.governance_status()

    def audit_events(self, limit=50):
        return self._events[-limit:]

    def respond(self, text):
        self._events.append({"action": "chat", "query": text})
        return f"echo:{text}"


class ApiServerTests(unittest.TestCase):
    def setUp(self):
        self.engine = _StubEngine()
        self.srv = ThreadedHTTPServer(("127.0.0.1", 0), ChatHandler)
        self.srv.engine = self.engine
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def _get_json(self, path):
        req = Request(f"http://127.0.0.1:{self.port}{path}", headers={"Accept": "application/json"})
        with urlopen(req, timeout=3) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_json(self, path, payload):
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=3) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_governance_get_and_post(self):
        before = self._get_json("/governance")
        self.assertEqual(before.get("autonomy_paused"), False)

        after = self._post_json("/governance", {"autonomy_paused": True, "autonomy_budget_remaining": 2})
        self.assertEqual(after.get("autonomy_paused"), True)
        self.assertEqual(after.get("autonomy_budget_remaining"), 2)

    def test_audit_endpoint(self):
        _ = self._post_json("/chat", {"text": "hello"})
        out = self._get_json("/audit")
        self.assertIn("events", out)
        self.assertTrue(len(out["events"]) >= 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
