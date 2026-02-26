import os
import shutil
import unittest

from core import trainer


class TrainerTests(unittest.TestCase):
    def setUp(self):
        # cleanup potential output directories
        for path in ("data/finetuned-model", "data/llm_finetuned"):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
