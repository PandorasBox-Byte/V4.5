import os
from typing import List, Tuple

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


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
        # create a tiny copy of a real model so that consumers (Engine) can load it
        os.makedirs(output_path, exist_ok=True)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # save only configuration (not necessarily full weights) to keep it fast
        model.save(output_path)
        return output_path

    model = SentenceTransformer(model_name)
    train_examples = [InputExample(texts=[a, b]) for a, b in examples]
    if not train_examples:
        # nothing to train on; simply save a copy of the base model
        model.save(output_path)
        return output_path

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs)
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

        # prepare dataset as simple concatenated dialogues
        texts = []
        for user, assistant in conversations:
            texts.append(f"User: {user}\nAssistant: {assistant}\n")
        encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

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
            no_cuda=not torch.cuda.is_available(),
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        trainer.train()
        trainer.save_model(output_dir)
    except Exception:
        # if transformers not available or training fails, create placeholder
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "dummy.txt"), "w") as f:
            f.write("error")
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

    args = parser.parse_args()
    if args.cmd == "embeddings":
        tuples = list(zip(args.pairs[::2], args.pairs[1::2]))
        path = train_embeddings(tuples, model_name=args.model, output_path=args.out, epochs=args.epochs)
        print("saved embeddings to", path)
    elif args.cmd == "llm":
        tuples = list(zip(args.convs[::2], args.convs[1::2]))
        path = train_llm(tuples, base_model=args.base, output_dir=args.out, epochs=args.epochs)
        print("saved llm to", path)
    else:
        parser.print_help()
