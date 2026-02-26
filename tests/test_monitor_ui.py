import threading
import time
import unittest

from core.monitor_ui import Dashboard


class MonitorUITests(unittest.TestCase):
    def test_dashboard_start_engine(self):
        # Dashboard.start_engine should spawn a thread that creates an Engine
        dash = Dashboard()
        t = dash.start_engine()
        self.assertTrue(isinstance(t, threading.Thread))
        # wait for engine thread to complete startup (join with timeout)
        t.join(timeout=5)
        self.assertTrue(dash.engine is not None and hasattr(dash.engine, 'status'))
        status = dash.engine.status()
        self.assertIsInstance(status, dict)
        self.assertIn("ready", status)

    def test_fetch_status_api_unreachable(self):
        dash = Dashboard(api_url="http://127.0.0.1:9999")
        status = dash._fetch_status()
        self.assertEqual(status, {"error": "unreachable"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
