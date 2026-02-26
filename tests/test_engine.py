import os
import unittest
import torch
import types


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
        # clear on-disk state so tests are isolated
        for fname in ("data/memory.json", "data/embeddings.pt"):
            try:
                os.remove(fname)
            except FileNotFoundError:
                pass

    def test_similarity_sequence(self):
        # default responder is simple
        engine = self.Engine()

        r1 = engine.respond("alpha")
        self.assertIn("I understand you", r1)

        r2 = engine.respond("beta")
        self.assertIn("I understand you", r2)

        r3 = engine.respond("alpha")
        self.assertTrue(
            ("You previously said something similar" in r3) or ("I understand you" in r3)
        )

    def test_smart_responder_memory_hint(self):
        os.environ["EVOAI_RESPONDER"] = "smart"
        engine = self.Engine()
        engine.respond("foo bar baz")
        response = engine.respond("foo bar baz")
        # second reply may be a memory hint or an avoidance message
        self.assertTrue(
            ("I recall you mentioned" in response)
            or ("already said" in response)
        )
        del os.environ["EVOAI_RESPONDER"]

    def test_plugin_knowledge(self):
        # write a temporary knowledge line and ensure plugin returns it
        os.environ["EVOAI_RESPONDER"] = "smart"
        path = os.path.join("data", "knowledge.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("The sky is blue\n")
        engine = self.Engine()
        # query something related to "sky" so plugin should match
        resp = engine.respond("what color is the sky?")
        self.assertIn("According to my notes", resp)
        # cleanup
        open(path, "w").close()
        del os.environ["EVOAI_RESPONDER"]

    def test_long_input_handling(self):
        engine = self.Engine()
        long_msg = "This is a long message to test handling. " * 40
        r = engine.respond(long_msg)
        self.assertIn("I understand you", r)

    def test_llm_integration(self):
        # create engine with smart responder then inject a dummy LLM
        os.environ["EVOAI_RESPONDER"] = "smart"
        engine = self.Engine()
        # stub model/tokenizer
        # model captures generation parameters
        captured = {}
        def fake_generate(**kwargs):
            captured.update(kwargs)
            return torch.tensor([[0]])
        engine.llm_model = types.SimpleNamespace(
            generate=fake_generate
        )
        class DummyTok:
            def __call__(self, prompt, return_tensors=None):
                return {"input_ids": torch.tensor([[0]])}
            def decode(self, ids, skip_special_tokens=True):
                return "llm reply"
        engine.llm_tokenizer = DummyTok()
        engine.llm_device = "cpu"

        resp = engine.respond("Hello")
        # should use the LLM instead of default text
        self.assertEqual(resp, "llm reply")
        # confirm generation params were passed through
        self.assertIn("max_new_tokens", captured)
        self.assertEqual(captured.get("max_new_tokens"), engine.llm_params["max_new_tokens"])
        del os.environ["EVOAI_RESPONDER"]

    def test_repetition_avoidance(self):
        os.environ["EVOAI_RESPONDER"] = "smart"
        engine = self.Engine()
        first = engine.respond("repeat me")
        second = engine.respond("repeat me")
        self.assertNotEqual(first, second)
        self.assertTrue("already" in second or "expand" in second)
        del os.environ["EVOAI_RESPONDER"]

    def test_repetition_with_llm(self):
        os.environ["EVOAI_RESPONDER"] = "smart"
        engine = self.Engine()
        # stub model that returns different text on subsequent calls
        counter = {"n": 0}
        def fake_generate(**kwargs):
            counter["n"] += 1
            return torch.tensor([[counter["n"]]])
        engine.llm_model = types.SimpleNamespace(generate=fake_generate)
        class DummyTok2:
            def __call__(self, prompt, return_tensors=None):
                return {"input_ids": torch.tensor([[0]])}
            def decode(self, ids, skip_special_tokens=True):
                return f"gen{ids.item()}"
        engine.llm_tokenizer = DummyTok2()
        engine.llm_device = "cpu"

        r1 = engine.respond("foo")
        r2 = engine.respond("foo")
        self.assertNotEqual(r1, r2)
        self.assertTrue(r2.startswith("gen"))
        del os.environ["EVOAI_RESPONDER"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
