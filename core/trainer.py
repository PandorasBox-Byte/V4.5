import os
from typing import List, Tuple
import json


def train_embeddings(
    examples: List[Tuple[str, str]],
    model_name: str = "all-MiniLM-L6-v2",
    output_path: str = "data/finetuned-model",
    epochs: int = 1,
) -> str:
    """Fine-tune a sentence-transformers model on a list of text pairs.

    ``examples`` should be an iterable of ``(text1, text2)`` tuples.  This
    uses a simple multiple-negatives ranking loss so that similar pairs are
    pulled together in embedding space.

    The returned value is the path where the model was saved (``output_path``).

    If ``model_name`` is ``"dummy"`` the function does not perform any real
    training; instead a placeholder directory is created so callers can still
    exercise the workflow in unit tests or on CI without incurring CPU/GPU
    cost.
    """
    if model_name == "dummy":
        # lightweight placeholder for test/CI runs without heavy ML deps
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.json"), "w", encoding="utf-8") as f:
            f.write('{"dummy": true}')
        return output_path

    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
    except Exception:
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.json"), "w", encoding="utf-8") as f:
            f.write('{"fallback": "missing_dependencies"}')
        return output_path

    model = SentenceTransformer(model_name)
    train_examples = [InputExample(texts=[a, b]) for a, b in examples]
    if not train_examples:
        # nothing to train on; simply save a copy of the base model
        model.save(output_path)
        return output_path

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    try:
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs)
    except Exception:
        model.save(output_path)
        return output_path
    model.save(output_path)
    return output_path


def train_llm(
    conversations: List[Tuple[str, str]],
    base_model: str = "gpt2",
    output_dir: str = "data/llm_finetuned",
    epochs: int = 1,
):
    """Lightweight fine-tuning of a causal LM on conversation pairs.

    ``conversations`` should be a list of ``(user, assistant)`` strings.  The
    data is concatenated with a simple separator and used to continue
    training the base model.

    The function is intentionally minimal and may not be suitable for
    production; it exists mainly to illustrate how one might adapt the
    workspace for further "training".  When ``base_model`` is ``"dummy"`` the
    routine writes a placeholder file to avoid heavy computation during tests.
    """
    if base_model == "dummy":
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "dummy.txt"), "w") as f:
            f.write("dummy")
        return output_dir

    # real training path (very small and CPU-friendly by default)
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        import torch

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        # prepare dataset as simple concatenated dialogues
        texts = []
        for user, assistant in conversations:
            texts.append(f"User: {user}\nAssistant: {assistant}\n")
        encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        encodings["labels"] = encodings["input_ids"].clone()

        class ConvDataset(torch.utils.data.Dataset):
            def __init__(self, enc):
                self.enc = enc
            def __len__(self):
                return self.enc.input_ids.size(0)
            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.enc.items()}

        dataset = ConvDataset(encodings)
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            logging_steps=10,
            save_steps=10,
            save_total_limit=1,
            use_cpu=not torch.cuda.is_available(),
            report_to=[],
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
    except Exception:
        # if transformers not available or training fails, create placeholder
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "dummy.txt"), "w") as f:
            f.write("error")
    return output_dir


def train_decision_policy(
    conversations: List[Tuple[str, str]],
    output_dir: str = "data/decision_policy",
    epochs: int = 30,
    width: int = 512,
    depth: int = 12,
) -> str:
    """Train the neural decision policy with synthetic supervision.

    The policy predicts routing actions used by Engine (delegate/clarify/llm/etc).
    If torch is unavailable, a fallback metadata file is written so startup remains stable.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from core.decision_policy import DecisionPolicy, _DecisionNet
    except Exception:
        with open(os.path.join(output_dir, "fallback.json"), "w", encoding="utf-8") as f:
            json.dump({"fallback": "missing_torch_or_policy_deps"}, f, indent=2)
        return output_dir

    policy = DecisionPolicy()

    class _FakeEngine:
        def __init__(self):
            self.last_user = ""
            self.plugins = [object()]
            self.llm_model = object()
            self.llm_tokenizer = object()
            self.memory = [{"user": "seed"}, {"assistant": "seed"}]
            self.corpus_texts = ["seed message"]
            self.similarity_threshold = 0.7
            self.embeddings_cache = [0.1]
            self.responder = type("SmartResponder", (), {})()

    def _label_for_text(text: str) -> int:
        t = (text or "").strip().lower()
        if not t:
            return DecisionPolicy.ACTIONS.index("simple_reply")
        if any(k in t.split() for k in ("it", "this", "that", "they")) and "?" in t:
            return DecisionPolicy.ACTIONS.index("clarify_first")
        if any(k in t for k in ("debug", "error", "traceback", "bug", "fix")):
            return DecisionPolicy.ACTIONS.index("delegate")
        if any(k in t for k in ("what next", "next", "proactive", "plan ahead")):
            return DecisionPolicy.ACTIONS.index("proactive_prompt")
        if "?" in t and any(k in t for k in ("why", "how", "can you", "should", "would")):
            return DecisionPolicy.ACTIONS.index("llm_generate")
        if len(t.split()) <= 3:
            return DecisionPolicy.ACTIONS.index("simple_reply")
        return DecisionPolicy.ACTIONS.index("memory_hint")

    # Build synthetic curriculum from user prompts + augmentations.
    rows: List[Tuple[list[float], int]] = []
    fake = _FakeEngine()
    prompts = [u for (u, _a) in conversations if isinstance(u, str)]
    for prompt in prompts:
        variants = {
            prompt,
            prompt.lower(),
            f"{prompt}?",
            f"Can you help me with: {prompt}",
            f"I got an error in: {prompt}",
            f"what should I do next about {prompt}",
        }
        for v in variants:
            feats = policy._feature_vector(v, fake)
            rows.append((feats, _label_for_text(v)))

    if not rows:
        rows.append(([0.0] * policy.input_dim, DecisionPolicy.ACTIONS.index("delegate")))

    x = torch.tensor([r[0] for r in rows], dtype=torch.float32)
    y = torch.tensor([r[1] for r in rows], dtype=torch.long)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=True)

    model = _DecisionNet(
        input_dim=policy.input_dim,
        width=width,
        depth=depth,
        action_dim=len(DecisionPolicy.ACTIONS),
        dropout=float(os.environ.get("EVOAI_DECISION_DROPOUT", "0.05")),
    )
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for _ in range(max(1, epochs)):
        for xb, yb in loader:
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

    model_path = os.path.join(output_dir, "model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": policy.input_dim,
            "width": width,
            "depth": depth,
            "epochs": epochs,
            "action_space": DecisionPolicy.ACTIONS,
            "samples": len(rows),
        },
        model_path,
    )

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dim": policy.input_dim,
                "width": width,
                "depth": depth,
                "epochs": epochs,
                "samples": len(rows),
                "actions": DecisionPolicy.ACTIONS,
                "model_path": model_path,
            },
            f,
            indent=2,
        )

    return output_dir


# Command-line entrypoints for manual training of embeddings or LLMs
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train EvoAI models")
    sub = parser.add_subparsers(dest="cmd")

    emb = sub.add_parser("embeddings", help="fine-tune embedding model")
    emb.add_argument("--pairs", nargs="*", help="user assistant pairs", default=[])
    emb.add_argument("--model", default="all-MiniLM-L6-v2")
    emb.add_argument("--out", default="data/finetuned-model")
    emb.add_argument("--epochs", type=int, default=1)

    llm = sub.add_parser("llm", help="fine-tune LLM")
    llm.add_argument("--convs", nargs="*", help="user assistant pairs", default=[])
    llm.add_argument("--base", default="gpt2")
    llm.add_argument("--out", default="data/llm_finetuned")
    llm.add_argument("--epochs", type=int, default=1)

    dec = sub.add_parser("decision", help="train decision policy")
    dec.add_argument("--pairs", nargs="*", help="user assistant pairs", default=[])
    dec.add_argument("--out", default="data/decision_policy")
    dec.add_argument("--epochs", type=int, default=30)
    dec.add_argument("--width", type=int, default=512)
    dec.add_argument("--depth", type=int, default=12)

    args = parser.parse_args()
    if args.cmd == "embeddings":
        tuples = list(zip(args.pairs[::2], args.pairs[1::2]))
        path = train_embeddings(tuples, model_name=args.model, output_path=args.out, epochs=args.epochs)
        print("saved embeddings to", path)
    elif args.cmd == "llm":
        tuples = list(zip(args.convs[::2], args.convs[1::2]))
        path = train_llm(tuples, base_model=args.base, output_dir=args.out, epochs=args.epochs)
        print("saved llm to", path)
    elif args.cmd == "decision":
        tuples = list(zip(args.pairs[::2], args.pairs[1::2]))
        path = train_decision_policy(
            tuples,
            output_dir=args.out,
            epochs=args.epochs,
            width=args.width,
            depth=args.depth,
        )
        print("saved decision policy to", path)
    else:
        parser.print_help()
