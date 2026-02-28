import os
import unittest
import types
import io
from contextlib import redirect_stdout
import json
from unittest.mock import patch


class _FakeValue:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def __getitem__(self, idx):
        value = self._values[idx]
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (int, float)):
            return _FakeValue(value[0])
        return value


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
        for fname in ("data/custom_conversations.json", "data/conversation_capture_meta.json"):
            try:
                os.remove(fname)
            except FileNotFoundError:
                pass

    def test_similarity_sequence(self):
        # default responder is simple
        engine = self.Engine()

        r1 = engine.respond("alpha")
        # first two replies should be non-empty and distinct
        self.assertTrue(r1)

        r2 = engine.respond("beta")
        self.assertTrue(r2)
        self.assertNotEqual(r1, r2)

        r3 = engine.respond("alpha")
        # should still mention "alpha" or similarity in some form
        self.assertTrue(
            "alpha" in r3.lower() or "similar" in r3.lower()
        )

    def test_smart_responder_memory_hint(self):
        os.environ["EVOAI_RESPONDER"] = "smart"
        engine = self.Engine()
        engine.respond("foo bar baz")
        response = engine.respond("foo bar baz")
        # second reply should not equal the first and should hint about memory
        self.assertNotEqual(response, "I understand you said: 'foo bar baz'")
        self.assertTrue("recall" in response.lower() or "already" in response.lower() or "earlier" in response.lower())
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
        # the plugin should mention "notes" in some form
        self.assertIn("notes", resp.lower())
        # cleanup
        open(path, "w").close()
        del os.environ["EVOAI_RESPONDER"]

    def test_long_input_handling(self):
        engine = self.Engine()
        long_msg = "This is a long message to test handling. " * 40
        r = engine.respond(long_msg)
        # engine should return something reasonable and not raise errors
        self.assertTrue(isinstance(r, str) and len(r) > 0)

    def test_llm_integration(self):
        # create engine with smart responder then inject a dummy LLM
        os.environ["EVOAI_RESPONDER"] = "smart"
        # disable thesaurus for deterministic output
        os.environ["EVOAI_USE_THESAURUS"] = "0"
        engine = self.Engine()
        # stub model/tokenizer
        # model captures generation parameters
        captured = {}
        def fake_generate(**kwargs):
            captured.update(kwargs)
            return _FakeTensor([[0]])
        engine.llm_model = types.SimpleNamespace(
            generate=fake_generate
        )
        class DummyTok:
            def __call__(self, prompt, return_tensors=None):
                return {"input_ids": [[0]]}
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
        os.environ.pop("EVOAI_USE_THESAURUS", None)

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
            return _FakeTensor([[counter["n"]]])
        engine.llm_model = types.SimpleNamespace(generate=fake_generate)
        class DummyTok2:
            def __call__(self, prompt, return_tensors=None):
                return {"input_ids": [[0]]}
            def decode(self, ids, skip_special_tokens=True):
                return f"gen{ids.item()}"
        engine.llm_tokenizer = DummyTok2()
        engine.llm_device = "cpu"

        r1 = engine.respond("foo")
        r2 = engine.respond("foo")
        self.assertNotEqual(r1, r2)
        self.assertTrue(r2.startswith("gen"))
        del os.environ["EVOAI_RESPONDER"]

    def test_language_enhancement_hook(self):
        # verify that replies are passed through language_utils.enhance_text
        engine = self.Engine()
        import core.language_utils as lu
        orig = lu.enhance_text
        lu.enhance_text = lambda t: t + "++"
        try:
            r = engine.respond("hello")
            self.assertTrue(r.endswith("++"))
        finally:
            lu.enhance_text = orig

    def test_clarification_prompt(self):
        engine = self.Engine()
        r = engine.respond("it is nice")
        # should ask a question mentioning 'it' and ending with a question mark
        self.assertIn("it", r.lower())
        self.assertIn("?", r)
        # subsequent call about same topic shouldn't repeat the question
        r2 = engine.respond("it is nice")
        self.assertNotIn("?", r2)

    def test_status_and_api(self):
        # engine.status should be a dict containing at least 'ready'
        engine = self.Engine()
        status = engine.status()
        self.assertIsInstance(status, dict)
        self.assertIn("ready", status)
        # start api server in background and query /status
        from core.api_server import run_server
        t = run_server(engine, addr="127.0.0.1", port=0)
        # run_server should return a thread object when start_thread=True
        self.assertTrue(hasattr(t, "is_alive"))

    def test_loader_mode_engine_init_is_quiet(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            _ = self.Engine(progress_cb=lambda f, m: None)
        self.assertEqual(buf.getvalue().strip(), "")

    def test_invalid_memory_file_does_not_crash(self):
        with open("data/memory.json", "w", encoding="utf-8") as f:
            f.write("{ not-valid-json")

        engine = self.Engine()
        self.assertEqual(engine.memory, [])
        self.assertTrue(engine.respond("hello"))

    def test_runtime_memory_and_corpus_stay_bounded(self):
        prev_max = os.environ.get("EVOAI_MAX_MEMORY")
        prev_thesaurus = os.environ.get("EVOAI_USE_THESAURUS")
        os.environ["EVOAI_MAX_MEMORY"] = "4"
        os.environ["EVOAI_USE_THESAURUS"] = "0"
        try:
            engine = self.Engine()
            for idx in range(12):
                engine.respond(f"message {idx}")

            self.assertLessEqual(len(engine.memory), 4)
            self.assertLessEqual(len(engine.corpus_texts), 2)
        finally:
            if prev_max is None:
                os.environ.pop("EVOAI_MAX_MEMORY", None)
            else:
                os.environ["EVOAI_MAX_MEMORY"] = prev_max
            if prev_thesaurus is None:
                os.environ.pop("EVOAI_USE_THESAURUS", None)
            else:
                os.environ["EVOAI_USE_THESAURUS"] = prev_thesaurus

    def test_decision_layer_status_fields_present(self):
        engine = self.Engine()
        status = engine.status()
        self.assertIn("decision_layer", status)
        self.assertIn("decision_depth", status)
        self.assertIn("decision_width", status)
        self.assertIn("safety_gate", status)
        self.assertIn("safety_allow_network", status)
        self.assertIn("safety_allow_self_modify", status)
        self.assertIn("autonomy_paused", status)
        self.assertIn("autonomy_budget_max", status)
        self.assertIn("autonomy_budget_remaining", status)
        self.assertIn("audit_events", status)
        self.assertIn("governance_policy_loaded", status)
        self.assertIn("governance_policy_recommendation", status)
        self.assertIn("governance_policy_applied", status)

    def test_governance_policy_tightens_budget_defaults(self):
        policy_dir = os.path.join("data", "governance_policy")
        os.makedirs(policy_dir, exist_ok=True)
        policy_path = os.path.join(policy_dir, "metadata.json")
        with open(policy_path, "w", encoding="utf-8") as f:
            json.dump({"recommendation": "tighten_autonomy"}, f)

        prev_autotune = os.environ.get("EVOAI_GOVERNANCE_POLICY_AUTOTUNE")
        prev_budget_max = os.environ.get("EVOAI_AUTONOMY_BUDGET_MAX")
        prev_budget_remaining = os.environ.get("EVOAI_AUTONOMY_BUDGET_REMAINING")
        prev_thes = os.environ.get("EVOAI_USE_THESAURUS")

        os.environ["EVOAI_GOVERNANCE_POLICY_AUTOTUNE"] = "1"
        os.environ.pop("EVOAI_AUTONOMY_BUDGET_MAX", None)
        os.environ.pop("EVOAI_AUTONOMY_BUDGET_REMAINING", None)
        os.environ["EVOAI_USE_THESAURUS"] = "0"

        try:
            engine = self.Engine()
            status = engine.status()
            self.assertEqual(status.get("governance_policy_loaded"), "true")
            self.assertEqual(status.get("governance_policy_recommendation"), "tighten_autonomy")
            self.assertEqual(status.get("governance_policy_applied"), "budget_tightened")
            self.assertLess(int(status.get("autonomy_budget_max", "0")), 25)
        finally:
            if prev_autotune is None:
                os.environ.pop("EVOAI_GOVERNANCE_POLICY_AUTOTUNE", None)
            else:
                os.environ["EVOAI_GOVERNANCE_POLICY_AUTOTUNE"] = prev_autotune

            if prev_budget_max is None:
                os.environ.pop("EVOAI_AUTONOMY_BUDGET_MAX", None)
            else:
                os.environ["EVOAI_AUTONOMY_BUDGET_MAX"] = prev_budget_max

            if prev_budget_remaining is None:
                os.environ.pop("EVOAI_AUTONOMY_BUDGET_REMAINING", None)
            else:
                os.environ["EVOAI_AUTONOMY_BUDGET_REMAINING"] = prev_budget_remaining

            if prev_thes is None:
                os.environ.pop("EVOAI_USE_THESAURUS", None)
            else:
                os.environ["EVOAI_USE_THESAURUS"] = prev_thes

    def test_governance_pause_blocks_autonomous_action(self):
        prev_modify = os.environ.get("EVOAI_ALLOW_SELF_MODIFY")
        prev_thes = os.environ.get("EVOAI_USE_THESAURUS")
        os.environ["EVOAI_ALLOW_SELF_MODIFY"] = "1"
        os.environ["EVOAI_USE_THESAURUS"] = "0"
        try:
            engine = self.Engine()
            engine.update_governance({"autonomy_paused": True})
            with patch.object(
                engine.decision_policy,
                "decide",
                return_value={"action": "tested_apply", "confidence": 1.0, "elapsed_ms": 1.0, "reason": "unit"},
            ):
                reply = engine.respond("rewrite code now")
            self.assertIn("governance blocked", reply.lower())
            self.assertIn("autonomy_paused", engine.status().get("safety_last_result", ""))
        finally:
            if prev_modify is None:
                os.environ.pop("EVOAI_ALLOW_SELF_MODIFY", None)
            else:
                os.environ["EVOAI_ALLOW_SELF_MODIFY"] = prev_modify
            if prev_thes is None:
                os.environ.pop("EVOAI_USE_THESAURUS", None)
            else:
                os.environ["EVOAI_USE_THESAURUS"] = prev_thes

    def test_governance_budget_exhaustion_blocks_second_action(self):
        prev_thes = os.environ.get("EVOAI_USE_THESAURUS")
        os.environ["EVOAI_USE_THESAURUS"] = "0"
        try:
            engine = self.Engine()
            engine.update_governance(
                {
                    "autonomy_paused": False,
                    "autonomy_budget_max": 1,
                    "autonomy_budget_remaining": 1,
                }
            )

            with patch.object(
                engine.decision_policy,
                "decide",
                return_value={"action": "code_intel_query", "confidence": 1.0, "elapsed_ms": 1.0, "reason": "unit"},
            ):
                first = engine.respond("analyze optimize path")
                second = engine.respond("analyze optimize path again")

            self.assertIn("code intel", first.lower())
            self.assertIn("governance blocked", second.lower())
            self.assertEqual(engine.status().get("autonomy_budget_remaining"), "0")
            self.assertIn("autonomy_budget_exhausted", engine.status().get("safety_last_result", ""))
        finally:
            if prev_thes is None:
                os.environ.pop("EVOAI_USE_THESAURUS", None)
            else:
                os.environ["EVOAI_USE_THESAURUS"] = prev_thes

    def test_safety_gate_blocks_external_backend(self):
        prev_net = os.environ.get("EVOAI_ALLOW_NETWORK_ACTIONS")
        prev_chat = os.environ.get("EVOAI_COPILOT_CHAT")
        prev_thes = os.environ.get("EVOAI_USE_THESAURUS")
        os.environ["EVOAI_ALLOW_NETWORK_ACTIONS"] = "0"
        os.environ["EVOAI_COPILOT_CHAT"] = "1"
        os.environ["EVOAI_USE_THESAURUS"] = "0"
        try:
            engine = self.Engine()

            class DummyExternal:
                def generate_sync(self, prompt, model=None, **kwargs):
                    return "External backend reply"

            engine.external_backend = DummyExternal()
            reply = engine.respond("please use internet research")
            status = engine.status()
            self.assertNotIn("External backend reply", reply)
            self.assertIn("blocked", status.get("safety_last_result", ""))
        finally:
            if prev_net is None:
                os.environ.pop("EVOAI_ALLOW_NETWORK_ACTIONS", None)
            else:
                os.environ["EVOAI_ALLOW_NETWORK_ACTIONS"] = prev_net
            if prev_chat is None:
                os.environ.pop("EVOAI_COPILOT_CHAT", None)
            else:
                os.environ["EVOAI_COPILOT_CHAT"] = prev_chat
            if prev_thes is None:
                os.environ.pop("EVOAI_USE_THESAURUS", None)
            else:
                os.environ["EVOAI_USE_THESAURUS"] = prev_thes

    def test_tested_apply_blocked_without_self_modify_permission(self):
        prev_modify = os.environ.get("EVOAI_ALLOW_SELF_MODIFY")
        prev_thes = os.environ.get("EVOAI_USE_THESAURUS")
        os.environ["EVOAI_ALLOW_SELF_MODIFY"] = "0"
        os.environ["EVOAI_USE_THESAURUS"] = "0"
        try:
            engine = self.Engine()
            with patch.object(
                engine.decision_policy,
                "decide",
                return_value={"action": "tested_apply", "confidence": 1.0, "elapsed_ms": 1.0, "reason": "unit"},
            ):
                reply = engine.respond("optimize code and rewrite now")
            self.assertIn("safe mode", reply.lower())
            self.assertIn("blocked", engine.status().get("safety_last_result", ""))
        finally:
            if prev_modify is None:
                os.environ.pop("EVOAI_ALLOW_SELF_MODIFY", None)
            else:
                os.environ["EVOAI_ALLOW_SELF_MODIFY"] = prev_modify
            if prev_thes is None:
                os.environ.pop("EVOAI_USE_THESAURUS", None)
            else:
                os.environ["EVOAI_USE_THESAURUS"] = prev_thes

    def test_tested_apply_scaffold_when_self_modify_allowed(self):
        prev_modify = os.environ.get("EVOAI_ALLOW_SELF_MODIFY")
        prev_thes = os.environ.get("EVOAI_USE_THESAURUS")
        os.environ["EVOAI_ALLOW_SELF_MODIFY"] = "1"
        os.environ["EVOAI_USE_THESAURUS"] = "0"
        try:
            engine = self.Engine()
            with patch.object(
                engine.decision_policy,
                "decide",
                return_value={"action": "tested_apply", "confidence": 1.0, "elapsed_ms": 1.0, "reason": "unit"},
            ):
                reply = engine.respond("rewrite code for speed")
            self.assertIn("manifest", reply.lower())
            self.assertEqual(engine.status().get("safety_last_result"), "ok")
            self.assertEqual(engine.status().get("tested_apply_last_ok"), "false")

            with patch.object(
                engine.tested_apply_orchestrator,
                "run",
                return_value={
                    "ok": True,
                    "summary": "Tested apply succeeded: 1 file(s) validated and applied.",
                    "reason": "ok",
                    "files": 1,
                    "retention_score": 0.82,
                },
            ):
                with patch.object(
                    engine.decision_policy,
                    "decide",
                    return_value={"action": "tested_apply", "confidence": 1.0, "elapsed_ms": 1.0, "reason": "unit"},
                ):
                    reply2 = engine.respond("rewrite code for speed")

            self.assertIn("succeeded", reply2.lower())
            self.assertEqual(engine.status().get("tested_apply_last_ok"), "true")
            self.assertEqual(engine.status().get("tested_apply_last_reason"), "ok")
            self.assertEqual(engine.status().get("tested_apply_last_files"), "1")
            self.assertEqual(engine.status().get("tested_apply_last_score"), "0.820")
        finally:
            if prev_modify is None:
                os.environ.pop("EVOAI_ALLOW_SELF_MODIFY", None)
            else:
                os.environ["EVOAI_ALLOW_SELF_MODIFY"] = prev_modify
            if prev_thes is None:
                os.environ.pop("EVOAI_USE_THESAURUS", None)
            else:
                os.environ["EVOAI_USE_THESAURUS"] = prev_thes

    def test_code_intel_query_returns_real_summary(self):
        prev_thes = os.environ.get("EVOAI_USE_THESAURUS")
        os.environ["EVOAI_USE_THESAURUS"] = "0"
        try:
            engine = self.Engine()
            with patch.object(
                engine.decision_policy,
                "decide",
                return_value={"action": "code_intel_query", "confidence": 1.0, "elapsed_ms": 1.0, "reason": "unit"},
            ):
                reply = engine.respond("find optimize function paths")
            self.assertIn("code intel", reply.lower())
            self.assertIn("hotspots", reply.lower())
            self.assertTrue(int(engine.status().get("code_intel_last_matches", "0")) >= 0)
        finally:
            if prev_thes is None:
                os.environ.pop("EVOAI_USE_THESAURUS", None)
            else:
                os.environ["EVOAI_USE_THESAURUS"] = prev_thes

    def test_research_query_uses_plugin_findings(self):
        prev_thes = os.environ.get("EVOAI_USE_THESAURUS")
        prev_web = os.environ.get("EVOAI_RESEARCH_ENABLE_WEB")
        prev_resp = os.environ.get("EVOAI_RESPONDER")
        os.environ["EVOAI_USE_THESAURUS"] = "0"
        os.environ["EVOAI_RESEARCH_ENABLE_WEB"] = "0"
        os.environ["EVOAI_RESPONDER"] = "smart"

        path = os.path.join("data", "knowledge.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("The sky is blue\n")

        try:
            engine = self.Engine()
            with patch.object(
                engine.decision_policy,
                "decide",
                return_value={"action": "research_query", "confidence": 1.0, "elapsed_ms": 1.0, "reason": "unit"},
            ):
                reply = engine.respond("Please research what color is the sky")
            self.assertIn("research:", reply.lower())
            self.assertIn("top result", reply.lower())
            self.assertTrue(int(engine.status().get("research_last_plugin_hits", "0")) >= 1)
        finally:
            open(path, "w").close()
            if prev_thes is None:
                os.environ.pop("EVOAI_USE_THESAURUS", None)
            else:
                os.environ["EVOAI_USE_THESAURUS"] = prev_thes
            if prev_web is None:
                os.environ.pop("EVOAI_RESEARCH_ENABLE_WEB", None)
            else:
                os.environ["EVOAI_RESEARCH_ENABLE_WEB"] = prev_web
            if prev_resp is None:
                os.environ.pop("EVOAI_RESPONDER", None)
            else:
                os.environ["EVOAI_RESPONDER"] = prev_resp

    def test_decision_layer_can_be_disabled(self):
        prev = os.environ.get("EVOAI_ENABLE_DECISION_LAYER")
        os.environ["EVOAI_ENABLE_DECISION_LAYER"] = "0"
        try:
            engine = self.Engine()
            status = engine.status()
            self.assertEqual(status.get("decision_layer"), "disabled")
            self.assertTrue(engine.respond("hello"))
        finally:
            if prev is None:
                os.environ.pop("EVOAI_ENABLE_DECISION_LAYER", None)
            else:
                os.environ["EVOAI_ENABLE_DECISION_LAYER"] = prev

    def test_decision_model_path_status_present(self):
        os.makedirs("data/decision_policy", exist_ok=True)
        with open("data/decision_policy/metadata.json", "w", encoding="utf-8") as f:
            json.dump({"ok": True}, f)
        prev = os.environ.get("EVOAI_DECISION_MODEL_PATH")
        os.environ["EVOAI_DECISION_MODEL_PATH"] = "data/decision_policy/model.pt"
        try:
            engine = self.Engine()
            status = engine.status()
            self.assertIn("decision_model_path", status)
            self.assertIn("decision_model_loaded", status)
        finally:
            if prev is None:
                os.environ.pop("EVOAI_DECISION_MODEL_PATH", None)
            else:
                os.environ["EVOAI_DECISION_MODEL_PATH"] = prev

    def test_external_backend_conversation_and_capture(self):
        prev_responder = os.environ.get("EVOAI_RESPONDER")
        prev_thesaurus = os.environ.get("EVOAI_USE_THESAURUS")
        prev_copilot = os.environ.get("EVOAI_COPILOT_CHAT")
        try:
            os.environ["EVOAI_RESPONDER"] = "smart"
            os.environ["EVOAI_USE_THESAURUS"] = "0"
            os.environ["EVOAI_COPILOT_CHAT"] = "1"

            engine = self.Engine()

            class DummyExternal:
                def generate_sync(self, prompt, model=None, **kwargs):
                    return "External detailed response with actionable debugging steps"

            engine.external_backend = DummyExternal()
            reply = engine.respond("Please help me debug this issue deeply")
            self.assertIn("External detailed response", reply)

            self.assertTrue(os.path.exists("data/custom_conversations.json"))
            with open("data/custom_conversations.json", "r", encoding="utf-8") as f:
                rows = json.load(f)
            self.assertTrue(any(isinstance(r, list) and len(r) == 2 for r in rows))

            self.assertTrue(os.path.exists("data/conversation_capture_meta.json"))
            with open("data/conversation_capture_meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertGreaterEqual(int(meta.get("new_samples_since_train", 0)), 1)
        finally:
            if prev_responder is None:
                os.environ.pop("EVOAI_RESPONDER", None)
            else:
                os.environ["EVOAI_RESPONDER"] = prev_responder
            if prev_thesaurus is None:
                os.environ.pop("EVOAI_USE_THESAURUS", None)
            else:
                os.environ["EVOAI_USE_THESAURUS"] = prev_thesaurus
            if prev_copilot is None:
                os.environ.pop("EVOAI_COPILOT_CHAT", None)
            else:
                os.environ["EVOAI_COPILOT_CHAT"] = prev_copilot

    def test_natural_language_self_test_triggers_checks(self):
        engine = self.Engine()
        with patch("core.self_repair.SelfRepair.run_tests", return_value=(True, "ok")) as mock_run_tests:
            response = engine.respond("can you test yourself and run diagnostics now?")
        mock_run_tests.assert_called_once()
        self.assertTrue("self-test" in response.lower() or "diagnostic" in response.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
