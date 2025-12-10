import argparse
import math
import os
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader

from config import ModelArgs
from model import Llama
from tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Compact LLaMA model on the HuggingFace FineWeb dataset."
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="sample-10BT",
        help="FineWeb subset to pull from HuggingFace. Defaults to the 10B token sample.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.01,
        help="Fraction of the dataset reserved for validation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the default batch size from ModelArgs.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Override the sequence length / block size from ModelArgs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train. Defaults to ModelArgs.epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=6e-4,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on. Use 'auto' to prefer CUDA when available.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit the number of training samples (useful for debugging).",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=2048,
        help="Limit the number of evaluation samples.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Log training metrics every N steps.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping value.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/fineweb",
        help="Where to store model checkpoints.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save a checkpoint every N epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available on this machine.")
    return device


def prepare_fineweb_splits(
    subset: str,
    val_split: float,
    seed: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
):
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name=subset,
        split="train",
        streaming=False,
    )

    splits = dataset.train_test_split(test_size=val_split, seed=seed)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]

    if max_train_samples is not None:
        max_train_samples = min(max_train_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(max_train_samples))

    if max_eval_samples is not None:
        max_eval_samples = min(max_eval_samples, len(eval_dataset))
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    return train_dataset, eval_dataset


class FineWebCollator:
    """Tokenizes a batch and builds shifted LM labels. Picklable for multi-worker DataLoader."""

    def __init__(self, tokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, str]]):
        texts = [sample["text"] for sample in batch]
        encodings = self.tokenizer(
            texts,
            max_length=self.block_size,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = encodings["input_ids"].clone()
        labels[:, :-1] = encodings["input_ids"][:, 1:]
        labels[:, -1] = self.eos_id
        encodings["labels"] = labels
        return encodings


def create_dataloaders(
    train_dataset,
    eval_dataset,
    tokenizer,
    block_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    collate_fn = FineWebCollator(tokenizer, block_size)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return train_loader, eval_loader


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=pad_token_id,
    )


@torch.no_grad()
def evaluate(model, dataloader, device: torch.device, pad_token_id: int) -> Dict[str, float]:
    model.eval()
    loss_total = 0.0
    steps = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"])
        loss = compute_loss(logits, batch["labels"], pad_token_id=pad_token_id)
        loss_total += loss.item()
        steps += 1

    model.train()
    avg_loss = loss_total / max(1, steps)
    return {"loss": avg_loss, "perplexity": math.exp(min(20, avg_loss))}


def save_checkpoint(model, optimizer, epoch: int, output_dir: str, args: argparse.Namespace):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"fineweb_epoch{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        checkpoint_path,
    )
    print(f"[Checkpoint] Saved model to {checkpoint_path}")


def train():
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = Tokenizer().ready_tokenizer()

    model_cfg = ModelArgs()
    if args.batch_size is not None:
        model_cfg.batch_size = args.batch_size
    if args.block_size is not None:
        model_cfg.block_size = args.block_size
    if args.epochs is not None:
        model_cfg.epochs = args.epochs
    model_cfg.device = str(device)

    train_dataset, eval_dataset = prepare_fineweb_splits(
        subset=args.subset,
        val_split=args.val_split,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    train_loader, eval_loader = create_dataloaders(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        block_size=model_cfg.block_size,
        batch_size=model_cfg.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    print(
        f"[Data] train={len(train_dataset)} examples "
        f"eval={len(eval_dataset)} examples "
        f"batch_size={model_cfg.batch_size}"
    )

    model = Llama(
        device=device,
        embeddings_dims=model_cfg.embeddings_dims,
        no_of_decoder_layers=model_cfg.no_of_decoder_layers,
        block_size=model_cfg.block_size,
        vocab_size=model_cfg.vocab_size,
        dropout=model_cfg.dropout,
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(model_cfg.beta_1, model_cfg.beta_2),
        eps=model_cfg.eps,
    )

    global_step = 0
    total_steps_per_epoch = len(train_loader)

    print(f"[DataLoader] steps_per_epoch={total_steps_per_epoch}")

    pad_token_id = tokenizer.pad_token_id

    for epoch in range(1, model_cfg.epochs + 1):
        print(f"[Epoch {epoch}] starting...")
        epoch_start = time.time()
        running_loss = 0.0
        window_steps = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(batch["input_ids"])
            loss = compute_loss(
                logits,
                batch["labels"],
                pad_token_id=pad_token_id,
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            window_steps += 1
            global_step += 1

            if step % args.log_every == 0 or step == total_steps_per_epoch:
                avg_loss = running_loss / max(1, window_steps)
                print(
                    f"[Epoch {epoch}/{model_cfg.epochs}] "
                    f"step {step}/{total_steps_per_epoch} "
                    f"loss {avg_loss:.4f}"
                )
                running_loss = 0.0
                window_steps = 0

        metrics = evaluate(
            model,
            eval_loader,
            device,
            pad_token_id=pad_token_id,
        )
        elapsed = time.time() - epoch_start
        print(
            f"[Eval] Epoch {epoch} | loss {metrics['loss']:.4f} "
            f"| ppl {metrics['perplexity']:.2f} | time {elapsed:.1f}s"
        )

        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, args.output_dir, args)


if __name__ == "__main__":
    train()
