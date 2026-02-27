from core import trainer


def main() -> None:
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
    ]

    out = trainer.train_llm(
        conversations,
        base_model="sshleifer/tiny-gpt2",
        output_dir="data/llm_finetuned",
        epochs=1,
    )
    print(f"LLM_TRAINED={out}")


if __name__ == "__main__":
    main()
