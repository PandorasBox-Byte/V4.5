import os
import unittest

from core.self_repair import SelfRepair


class SelfRepairTests(unittest.TestCase):
    def test_startup_self_test_can_be_disabled(self):
        prev = os.environ.get("EVOAI_STARTUP_SELF_TEST")
        os.environ["EVOAI_STARTUP_SELF_TEST"] = "0"
        try:
            ok, out = SelfRepair.run_tests(mode="startup", include_pytest=False)
            self.assertTrue(ok)
            self.assertIn("disabled", out.lower())
        finally:
            if prev is None:
                os.environ.pop("EVOAI_STARTUP_SELF_TEST", None)
            else:
                os.environ["EVOAI_STARTUP_SELF_TEST"] = prev

    def test_startup_self_test_reports_checks(self):
        prev = os.environ.get("EVOAI_STARTUP_SELF_TEST")
        os.environ["EVOAI_STARTUP_SELF_TEST"] = "1"
        seen = {}

        def _progress(frac, msg, check_name=None, passed=None):
            if check_name is not None and passed is not None:
                seen[str(check_name)] = bool(passed)

        try:
            ok, _out = SelfRepair.run_tests(
                progress_cb=_progress,
                mode="startup",
                include_pytest=False,
            )
            self.assertTrue(ok)
            self.assertIn("runtime", seen)
            self.assertIn("memory", seen)
            self.assertIn("embeddings", seen)
        finally:
            if prev is None:
                os.environ.pop("EVOAI_STARTUP_SELF_TEST", None)
            else:
                os.environ["EVOAI_STARTUP_SELF_TEST"] = prev


if __name__ == "__main__":
    unittest.main(verbosity=2)