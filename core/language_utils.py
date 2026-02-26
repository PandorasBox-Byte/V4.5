import random
import nltk
from nltk.corpus import wordnet as wn


def ensure_wordnet():
    """Make sure WordNet data is available, downloading if necessary."""
    try:
        # simple access will raise LookupError if not present
        wn.synsets("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


def get_synonyms(word: str) -> list[str]:
    """Return a list of synonyms for *word* using WordNet."""
    ensure_wordnet()
    syns = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            syns.add(lemma.name().replace("_", " "))
    return sorted(syns)


def enhance_text(text: str) -> str:
    """Make *text* more varied by replacing a few words with synonyms.

    This is intentionally simple: pick random nouns/adjectives/verbs and
    replace them with a randomly chosen synonym (if one exists).  The goal is
    to give replies a bit more variety and vocabulary without altering the
    meaning drastically.
    """
    ensure_wordnet()
    words = text.split()
    output = []
    for w in words:
        base = w.strip(".,!?;")
        # only try to replace short words
        if not base or len(base) < 3:
            output.append(w)
            continue
        syns = get_synonyms(base)
        # drop the word itself from candidates
        syns = [s for s in syns if s.lower() != base.lower()]
        if syns and random.random() < 0.3:
            choice = random.choice(syns)
            # reattach punctuation
            neww = w.replace(base, choice)
            output.append(neww)
        else:
            output.append(w)
    return " ".join(output)


def clarify_if_ambiguous(text: str) -> tuple[str | None, str | None]:
    """Return a clarification question and a "topic" ID if *text* seems
    ambiguous.

    The heuristics are simple: if the user used a pronoun or a generic word
    (``it``, ``that``, ``thing`` etc.) the engine isn't sure what is meant. The
    returned ``topic`` can be used by callers to avoid asking the same
    question repeatedly.
    """
    ambiguous_tokens = {"it", "this", "that", "they", "things", "thing", "something", "someone", "there"}
    words = text.lower().split()
    for w in words:
        if w in ambiguous_tokens:
            return (f"Could you clarify what you mean by '{w}'?", w)
    # additionally, if the sentence contains a word with many synonyms, it
    # might be vague; pick the first such word that has >5 synonyms.
    for w in words:
        syns = get_synonyms(w)
        if len(syns) > 5:
            return (f"When you say '{w}', are you referring to anything specific?", w)
    return (None, None)
