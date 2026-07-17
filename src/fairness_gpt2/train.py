"""Training for the three modes in Section 5.3: baseline / cda / cda_reg.

Usage
-----
uv run fairness-train --mode cda_reg --train data/quora-train.csv \
    --dev data/quora-dev.csv --out checkpoints/cda_reg --epochs 10
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import Collator, QQPDataset, build_counterfactual_pool, load_qqp
from .evaluate import evaluate
from .model import GPT2ParaphraseClassifier, build_tokenizer

SEED = 11711


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """L_fairness = 0.5 * (KL(p_cf || p_orig) + KL(p_orig || p_cf))  (Section 4.5)."""
    log_pa = F.log_softmax(logits_a, dim=-1)
    log_pb = F.log_softmax(logits_b, dim=-1)
    pa, pb = log_pa.exp(), log_pb.exp()
    kl_ab = (pa * (log_pa - log_pb)).sum(-1)
    kl_ba = (pb * (log_pb - log_pa)).sum(-1)
    return 0.5 * (kl_ab + kl_ba).mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "cda", "cda_reg"], required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)  # effective batch 64
    ap.add_argument("--lambda-fair", type=float, default=0.5)
    ap.add_argument("--cda-prob", type=float, default=0.5)
    ap.add_argument("--cf-per-epoch", type=int, default=1000)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument(
        "--eval-subset",
        type=int,
        default=0,
        help="Evaluate on only the first N dev pairs (0 = all).",
    )
    ap.add_argument(
        "--train-subset",
        type=int,
        default=0,
        help="Train on only the first N pairs (0 = all). Useful for smoke tests.",
    )
    ap.add_argument(
        "--save-half",
        action="store_true",
        help="Store fp16 weights (~250MB) for Streamlit Cloud deployment.",
    )
    args = ap.parse_args()

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    train_pairs = load_qqp(args.train)
    dev_pairs = load_qqp(args.dev)
    if args.train_subset:
        train_pairs = train_pairs[: args.train_subset]
    eval_pairs = dev_pairs[: args.eval_subset] if args.eval_subset else dev_pairs
    if device == "cpu":
        print(
            "\n  WARNING: no GPU detected — training on CPU.\n"
            "  A full run will take days. On Colab: Runtime > Change runtime type > T4 GPU.\n"
            "  If you have a GPU but see this, `uv run` may have reverted CUDA torch to the\n"
            "  CPU wheel pinned in uv.lock. Set UV_NO_SYNC=1 (see notebooks/train_qqp_colab.ipynb).\n"
        )
    print(
        f"train={len(train_pairs)}  eval={len(eval_pairs)}"
        f"{f' (subset of {len(dev_pairs)})' if args.eval_subset else ''}  device={device}"
    )

    tokenizer = build_tokenizer()
    model = GPT2ParaphraseClassifier().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    use_cda = args.mode in ("cda", "cda_reg")
    use_reg = args.mode == "cda_reg"
    collate = Collator(tokenizer, args.max_length, with_cf=use_reg)

    history = []
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(args.epochs):
        # Section 5.3: 1,000 fresh counterfactual pairs are added each epoch.
        epoch_pairs = list(train_pairs)
        if use_cda and args.cf_per_epoch:
            epoch_pairs += build_counterfactual_pool(
                train_pairs, n=args.cf_per_epoch, seed=SEED + epoch
            )

        ds = QQPDataset(
            epoch_pairs,
            cda_prob=args.cda_prob if use_cda else 0.0,
            return_counterfactual=use_reg,
            seed=SEED + epoch,
        )
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=2,
            drop_last=True,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        sum_task = sum_fair = 0.0
        steps = 0

        for i, batch in enumerate(tqdm(dl, desc=f"epoch {epoch}")):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(ids, mask)
                task_loss = F.cross_entropy(logits, labels)
                fair_loss = torch.zeros((), device=device)

                if use_reg:
                    cf_logits = model(
                        batch["cf_input_ids"].to(device),
                        batch["cf_attention_mask"].to(device),
                    )
                    fair_loss = symmetric_kl(cf_logits, logits)

                # L_total = L_task + lambda * L_fairness
                loss = (task_loss + args.lambda_fair * fair_loss) / args.grad_accum

            scaler.scale(loss).backward()
            sum_task += task_loss.item()
            sum_fair += fair_loss.detach().item()
            steps += 1

            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        metrics = evaluate(
            model,
            tokenizer,
            eval_pairs,
            device,
            batch_size=args.batch_size * 2,
            max_length=args.max_length,
        )
        row = {
            "epoch": epoch,
            "task_loss": round(sum_task / max(steps, 1), 4),
            "fairness_loss": round(sum_fair / max(steps, 1), 4),
            "dev_acc": round(metrics["accuracy"], 4),
            "subgroup_gap": round(metrics["subgroup_gap"], 4),
            "flip_rate": round(metrics["flip_rate"], 4),
        }
        history.append(row)
        print(json.dumps(row))

        model.save(args.out, half=args.save_half)
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump({"mode": args.mode, "args": vars(args), "history": history}, f, indent=2)

    # Publish final metrics where the dashboard looks for reproduced results.
    os.makedirs("results/reproduced", exist_ok=True)
    with open(f"results/reproduced/{args.mode}.json", "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "accuracy": metrics["accuracy"],
                "subgroup_gap": metrics["subgroup_gap"],
                "flip_rate": metrics["flip_rate"],
                "flips": metrics["flips"],
                "n_identity": metrics["n_identity"],
                "per_subgroup": metrics["per_subgroup"],
                "epochs": args.epochs,
                "seed": SEED,
            },
            f,
            indent=2,
        )

    tokenizer.save_pretrained(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()