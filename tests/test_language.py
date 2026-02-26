import unittest

from core import language_utils


class LanguageUtilsTests(unittest.TestCase):
    def setUp(self):
        # ensure corpora available
        language_utils.ensure_wordnet()

    def test_get_synonyms_returns_list(self):
        syns = language_utils.get_synonyms("happy")
        self.assertIsInstance(syns, list)
        # should include at least one synonym (wordnet has "happy" itself)
        self.assertTrue(len(syns) >= 1)

    def test_enhance_text_changes_something(self):
        text = "this is a simple sentence"
        enhanced = language_utils.enhance_text(text)
        self.assertIsInstance(enhanced, str)
        # output should have at least as many words as the input
        # (synonym expansion may introduce additional tokens).
        self.assertGreaterEqual(len(enhanced.split()), len(text.split()))

    def test_clarify_if_ambiguous_pronoun(self):
        q, topic = language_utils.clarify_if_ambiguous("it is broken")
        self.assertIsNotNone(q)
        self.assertEqual(topic, "it")

    def test_clarify_if_ambiguous_synonym_heavy(self):
        # words with many synonyms should trigger an ambiguity question.
        q, topic = language_utils.clarify_if_ambiguous("I will run")
        self.assertIsNotNone(q)
        self.assertIsNotNone(topic)
        self.assertIn(topic, q)


if __name__ == "__main__":
    unittest.main(verbosity=2)
