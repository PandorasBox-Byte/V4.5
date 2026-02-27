import argparse
import json
import os
import sys

if __name__ == "__main__" and __package__ is None:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_this_dir)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

from core import trainer


def _load_json(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _profile_defaults() -> dict:
    return {
        "name": "default",
        "tone": "concise, direct, friendly",
        "focus": ["clarity", "actionable next steps", "debugging support"],
        "do": [
            "give short actionable answers first",
            "offer step-by-step help when asked",
            "adapt depth based on user request",
        ],
        "avoid": [
            "rambling responses",
            "vague advice without concrete next steps",
        ],
    }


def _profile_to_examples(profile: dict) -> list[tuple[str, str]]:
    tone = profile.get("tone", "concise, direct, friendly")
    focus = ", ".join(profile.get("focus", []))
    do_rules = "; ".join(profile.get("do", []))
    avoid_rules = "; ".join(profile.get("avoid", []))

    return [
        (
            "How should you respond to me?",
            f"I will respond in a {tone} style with emphasis on {focus}.",
        ),
        (
            "What communication style should you use?",
            f"I should: {do_rules}. I should avoid: {avoid_rules}.",
        ),
        (
            "Can you adapt to my preference?",
            f"Yes. I will match your preferred tone: {tone}, while staying practical.",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EvoAI personalization datasets")
    parser.add_argument("--emb-epochs", type=int, default=2)
    parser.add_argument("--llm-epochs", type=int, default=2)
    parser.add_argument("--decision-epochs", type=int, default=40)
    parser.add_argument("--decision-width", type=int, default=512)
    parser.add_argument("--decision-depth", type=int, default=12)
    parser.add_argument("--emb-out", default="data/finetuned-model")
    parser.add_argument("--llm-out", default="data/llm_finetuned")
    parser.add_argument("--decision-out", default="data/decision_policy")
    parser.add_argument("--profile", default="data/personalization_profile.json")
    parser.add_argument("--extra-conversations", default="data/custom_conversations.json")
    parser.add_argument("--extra-pairs", default="data/custom_pairs.json")
    args = parser.parse_args()

    embedding_pairs = [
        ("hello", "hi"),
        ("goodbye", "see you later"),
        ("thanks", "thank you"),
        ("please help me", "can you help me"),
        ("i am confused", "i do not understand"),
        ("explain this", "can you clarify this"),
        ("summarize this", "give me a short summary"),
        ("what does this mean", "what is the meaning of this"),
        ("fix this bug", "help me debug this"),
        ("my code is broken", "the program is failing"),
        ("it crashes", "the app keeps crashing"),
        ("improve performance", "make this run faster"),
        ("optimize this", "improve efficiency"),
        ("write tests", "create unit tests"),
        ("add documentation", "write docs"),
        ("refactor this", "clean up this code"),
        ("how are you", "how are you doing"),
        ("what can you do", "what are your capabilities"),
        ("teach me", "help me learn"),
        ("give examples", "show me examples"),
        ("be concise", "keep it short"),
        ("be detailed", "explain in detail"),
        ("step by step", "walk me through it"),
        ("i need a plan", "help me plan this"),
        ("i made a mistake", "i did something wrong"),
        ("try again", "please retry"),
        ("that is unclear", "this is confusing"),
        ("what is next", "what should i do next"),
        ("can you rephrase", "say that differently"),
        ("understand my intent", "figure out what i mean"),
        ("i need help fast", "help me quickly"),
        ("walk me through this", "guide me step by step"),
        ("i feel stuck", "i am blocked"),
        ("explain like i am a beginner", "give me a beginner-friendly explanation"),
        ("how do i start", "what is the first step"),
        ("can you check this", "please review this"),
        ("is this correct", "does this look right"),
        ("what is the bug", "where is the issue"),
        ("i need troubleshooting", "help me diagnose this"),
        ("this is too long", "shorten this"),
        ("go deeper", "provide more depth"),
        ("can you simplify", "make this easier to understand"),
        ("what are my options", "what choices do i have"),
        ("what is the best approach", "which approach should i choose"),
        ("what are the tradeoffs", "compare pros and cons"),
        ("please give me a checklist", "provide a checklist"),
        ("what should i test", "how do i test this"),
        ("why did this fail", "what caused this failure"),
        ("let us fix it", "help me resolve it"),
        ("try a different way", "use another approach"),
        ("i need confidence", "how can i verify this works"),
        ("can you summarize decisions", "recap the decisions"),
        ("what did we learn", "key takeaways"),
        ("help me communicate this", "how do i explain this to my team"),
        ("turn this into tasks", "break this into action items"),
    ]

    conversations = [
        ("Hi", "Hello! How can I help you today?"),
        ("Can you help me with Python?", "Absolutely. Tell me your goal and I will guide you step by step."),
        ("I am new to coding", "Great start. We can go slowly and build confidence with small examples."),
        ("I got an error", "Share the full error message and the code around it; I will help pinpoint the cause."),
        ("Please explain simply", "Sure. I will keep it simple and practical."),
        ("Can you be concise?", "Yes. I will keep replies short and focused."),
        ("Can you be more detailed?", "Yes. I will provide deeper explanations and examples."),
        ("I am frustrated", "That is understandable. Let us break the problem into one small step first."),
        ("I do not understand this function", "No problem. I will explain what it does, inputs, outputs, and an example."),
        ("What should I do next?", "Next, run the smallest test that reproduces the issue and share the result."),
        ("Can you review my code?", "Yes. Paste the relevant file and I will check correctness, readability, and edge cases."),
        ("How can I improve this?", "I will suggest the highest-impact improvements first and keep changes minimal."),
        ("Can you summarize?", "Sure. I will summarize key points and give a clear action list."),
        ("What does this word mean in programming?", "I can define it in plain language and provide a concrete example."),
        ("Teach me debugging", "Start by reproducing consistently, isolate variables, then validate one hypothesis at a time."),
        ("I am overwhelmed", "Totally fair. Let us focus on one small, high-impact next step."),
        ("Can you give me a checklist?", "Yes. I will give you a short, practical checklist you can follow immediately."),
        ("I need this done today", "Understood. I will prioritize fast, reliable steps and defer non-essential work."),
        ("What should I do first?", "First, reproduce the issue consistently and capture the exact error output."),
        ("I do not know what to ask", "No problem. Tell me the outcome you want, and I will ask the right questions."),
        ("Please keep this brief", "Got it. I will keep responses short and actionable."),
        ("Can you think out loud less?", "Yes. I will focus on direct answers and clear actions."),
        ("Can you compare two options?", "Absolutely. I will compare tradeoffs and recommend one with reasons."),
        ("How do I avoid this next time?", "I will suggest a prevention plan: tests, checks, and process improvements."),
        ("Can you rewrite this better?", "Yes. Share the text and I will rewrite it clearly for your audience."),
        ("I need a commit message", "I can draft a concise commit message with scope and rationale."),
        ("Can you review this PR?", "Yes. I will check correctness, risk, readability, and missing tests."),
        ("How do I debug faster?", "Use a tight loop: reproduce, isolate, test one hypothesis, and log findings."),
        ("Please explain this error", "Share the full traceback and nearby code; I will translate it into plain language."),
        ("Can we plan this feature?", "Yes. I will break it into milestones with clear acceptance criteria."),
        ("I need help communicating to my team", "I can draft a clear summary with decisions, risks, and next steps."),
        ("Can you challenge my plan?", "Sure. I will point out assumptions, risks, and safer alternatives."),
        ("I made changes and now tests fail", "Let us isolate the smallest failing case and trace the regression source."),
        ("How can I make this more maintainable?", "I will suggest small refactors that improve clarity and reduce coupling."),
        ("What is the next best action?", "I will recommend one concrete next action with the highest impact."),
        ("Please be kind but direct", "Absolutely. I will be supportive, clear, and straightforward."),
        ("I need this in steps", "I will provide numbered steps and stop points for validation."),
        ("I want confidence before deploy", "I will give you a pre-deploy checklist and rollback safeguards."),
        ("Can you summarize in three bullets?", "Yes. I will provide exactly three concise bullets."),
        ("What should we automate?", "I will identify repetitive tasks and recommend high-value automation first."),
        ("Can you help me learn this deeply?", "Yes. I will start simple, then layer deeper concepts with examples."),
        ("What did we miss?", "I will scan for blind spots: edge cases, failure modes, and assumptions."),
        ("I am not sure this is right", "Let us verify with targeted tests and expected outcomes."),
        ("Can you adapt to my style?", "Yes. Tell me your preferred tone and depth, and I will match it."),
        ("Let us continue from where we left off", "Great. I will recap the last step and propose the next one."),
    ]

    profile = _load_json(args.profile) or _profile_defaults()
    conversations.extend(_profile_to_examples(profile))

    extra_pairs = _load_json(args.extra_pairs)
    if isinstance(extra_pairs, list):
        for row in extra_pairs:
            if isinstance(row, list) and len(row) == 2:
                embedding_pairs.append((str(row[0]), str(row[1])))

    extra_conversations = _load_json(args.extra_conversations)
    if isinstance(extra_conversations, list):
        for row in extra_conversations:
            if isinstance(row, list) and len(row) == 2:
                conversations.append((str(row[0]), str(row[1])))

    emb_path = trainer.train_embeddings(
        embedding_pairs,
        model_name="all-MiniLM-L6-v2",
        output_path=args.emb_out,
        epochs=args.emb_epochs,
    )
    print(f"EMB_TRAINED={emb_path}")

    llm_path = trainer.train_llm(
        conversations,
        base_model="sshleifer/tiny-gpt2",
        output_dir=args.llm_out,
        epochs=args.llm_epochs,
    )
    print(f"LLM_TRAINED={llm_path}")

    decision_path = trainer.train_decision_policy(
        conversations,
        output_dir=args.decision_out,
        epochs=args.decision_epochs,
        width=args.decision_width,
        depth=args.decision_depth,
    )
    print(f"DECISION_TRAINED={decision_path}")


if __name__ == "__main__":
    main()
