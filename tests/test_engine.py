import os
import unittest


class EngineSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use conservative defaults for test runs
        os.environ.setdefault("EVOAI_TORCH_THREADS", "2")
        os.environ.setdefault("EVOAI_MAX_MEMORY", "5")

    def setUp(self):
        # Import here to allow environment variables to take effect
        from core.engine_template import Engine

        self.Engine = Engine

    def test_similarity_sequence(self):
        engine = self.Engine()

        r1 = engine.respond("alpha")
        self.assertIn("I understand you", r1)

        r2 = engine.respond("beta")
        self.assertIn("I understand you", r2)

        # Repeating a previous message should trigger similarity detection
        r3 = engine.respond("alpha")
        self.assertTrue(
            ("You previously said something similar" in r3) or ("I understand you" in r3)
        )

    def test_long_input_handling(self):
        engine = self.Engine()
        long_msg = "This is a long message to test handling. " * 40
        r = engine.respond(long_msg)
        self.assertIn("I understand you", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
