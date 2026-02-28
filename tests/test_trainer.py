import os
import shutil
import unittest
import json

from core import trainer


class TrainerTests(unittest.TestCase):
    def setUp(self):
        # cleanup potential output directories
        for path in ("data/finetuned-model", "data/llm_finetuned", "data/decision_policy", "data/governance_policy"):
            if os.path.isdir(path):
                shutil.rmtree(path)

    def test_train_embeddings_dummy(self):
        out = trainer.train_embeddings([("a", "b")], model_name="dummy", output_path="data/finetuned-model")
        self.assertTrue(os.path.isdir(out))
        # ensure the saved directory looks like a SentenceTransformer model
        self.assertTrue(os.path.exists(os.path.join(out, "config.json")))

    def test_engine_loads_finetuned(self):
        # train dummy embeddings and then instruct Engine to load them
        path = trainer.train_embeddings([], model_name="dummy", output_path="data/finetuned-model")
        os.environ["EVOAI_FINETUNED_MODEL"] = path
        from core.engine_template import Engine
        e = Engine()
        # model should be a SentenceTransformer instance; nothing else to check
        self.assertTrue(hasattr(e, "model"))
        del os.environ["EVOAI_FINETUNED_MODEL"]

    def test_train_llm_dummy(self):
        out = trainer.train_llm([("hi", "hello")], base_model="dummy", output_dir="data/llm_finetuned")
        self.assertTrue(os.path.isdir(out))
        self.assertTrue(os.path.exists(os.path.join(out, "dummy.txt")))

    def test_train_decision_policy_outputs_artifacts(self):
        out = trainer.train_decision_policy(
            [("how do i debug this", "Start with the error output")],
            output_dir="data/decision_policy",
            epochs=2,
            width=64,
            depth=4,
        )
        self.assertTrue(os.path.isdir(out))
        self.assertTrue(
            os.path.exists(os.path.join(out, "metadata.json"))
            or os.path.exists(os.path.join(out, "fallback.json"))
        )

    def test_train_governance_policy_outputs_artifacts(self):
        os.makedirs("data", exist_ok=True)
        outcomes = os.path.join("data", "autonomy_outcomes.jsonl")
        with open(outcomes, "w", encoding="utf-8") as f:
            f.write('{"kind":"allow","action":"code_intel_query","reason":"ok"}\n')
            f.write('{"kind":"outcome","action":"tested_apply","reason":"ok","meta":{"ok":true,"retention_score":0.82}}\n')
            f.write('{"kind":"blocked","action":"tested_apply","reason":"autonomy_budget_exhausted"}\n')

        out = trainer.train_governance_policy(
            outcomes_path=outcomes,
            output_dir="data/governance_policy",
        )
        self.assertTrue(os.path.isdir(out))
        self.assertTrue(os.path.exists(os.path.join(out, "metadata.json")))
        with open(os.path.join(out, "metadata.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertIn("action_stats", meta)
        self.assertIn("avg_retention_score", meta)


if __name__ == "__main__":
    unittest.main(verbosity=2)
