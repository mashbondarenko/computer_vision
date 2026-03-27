import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from data import load_digits_splits
from model import VisionTransformer

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def eval_model(model: VisionTransformer, dataset, batch_size: int = 2048):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    all_pred, all_true, all_logits = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss_sum += criterion(logits, yb).item() * yb.numel()
            pred = logits.argmax(1)
            total += yb.numel()
            correct += (pred == yb).sum().item()
            all_pred.append(pred.cpu())
            all_true.append(yb.cpu())
            all_logits.append(logits.cpu())
    return {
        "loss": loss_sum / total,
        "acc": correct / total,
        "pred": torch.cat(all_pred).numpy(),
        "true": torch.cat(all_true).numpy(),
        "logits": torch.cat(all_logits).numpy(),
    }

def count_params(model: VisionTransformer) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def attention_cost(model: VisionTransformer, batch_size: int = 64, dtype_bytes: int = 4):
    seq_len = model.seq_len
    dim = model.embed_dim
    heads = model.heads
    depth = model.depth
    per_layer_attention_macs = 2 * seq_len * seq_len * dim
    per_layer_attn_matrix_elems = heads * seq_len * seq_len
    per_layer_attn_matrix_mb = per_layer_attn_matrix_elems * dtype_bytes / (1024 ** 2)
    return {
        "seq_len": seq_len,
        "per_layer_attention_macs": per_layer_attention_macs,
        "total_attention_macs": per_layer_attention_macs * depth,
        "per_layer_attn_matrix_mb_b1": per_layer_attn_matrix_mb,
        "all_layers_attn_matrix_mb_batch64": per_layer_attn_matrix_mb * depth * batch_size,
    }

def train_config(name: str, cfg: dict, train_ds, val_ds, test_ds, epochs: int, lr: float, seed: int):
    set_seed(seed)
    model = VisionTransformer(**cfg)
    train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    best_val = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        train_metrics = eval_model(model, train_ds)
        val_metrics = eval_model(model, val_ds)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
            }
        )
        if val_metrics["acc"] > best_val:
            best_val = val_metrics["acc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    train_metrics = eval_model(model, train_ds)
    val_metrics = eval_model(model, val_ds)
    test_metrics = eval_model(model, test_ds)

    result = {
        "name": name,
        "config": cfg,
        "params": count_params(model),
        "seq_len": model.seq_len,
        "train_acc": train_metrics["acc"],
        "val_acc": val_metrics["acc"],
        "test_acc": test_metrics["acc"],
        "train_loss": train_metrics["loss"],
        "val_loss": val_metrics["loss"],
        "test_loss": test_metrics["loss"],
        "cost": attention_cost(model, batch_size=64),
        "history": history,
    }
    return model, result, test_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    train_ds, val_ds, test_ds = load_digits_splits(seed=args.seed)

    configs = {
        "A_patch2_depth2_cls": dict(image_size=8, patch_size=2, embed_dim=32, depth=2, heads=4, use_cls_token=True, dropout=0.0),
        "B_patch4_depth2_cls": dict(image_size=8, patch_size=4, embed_dim=32, depth=2, heads=4, use_cls_token=True, dropout=0.0),
        "C_patch2_depth4_cls": dict(image_size=8, patch_size=2, embed_dim=32, depth=4, heads=4, use_cls_token=True, dropout=0.0),
        "D_patch2_depth2_mean": dict(image_size=8, patch_size=2, embed_dim=32, depth=2, heads=4, use_cls_token=False, dropout=0.0),
    }

    results = {}
    rows = []
    for name, cfg in configs.items():
        model, result, test_metrics = train_config(name, cfg, train_ds, val_ds, test_ds, args.epochs, args.lr, args.seed)
        results[name] = result
        rows.append(
            {
                "config": name,
                "patch_size": result["config"]["patch_size"],
                "depth": result["config"]["depth"],
                "use_cls_token": result["config"]["use_cls_token"],
                "embed_dim": result["config"]["embed_dim"],
                "params": result["params"],
                "seq_len": result["seq_len"],
                "val_acc": result["val_acc"],
                "test_acc": result["test_acc"],
                "attn_total_MACs": result["cost"]["total_attention_macs"],
                "attn_matrix_MB_per_layer_b1": result["cost"]["per_layer_attn_matrix_mb_b1"],
                "attn_matrix_MB_all_layers_batch64": result["cost"]["all_layers_attn_matrix_mb_batch64"],
            }
        )
        torch.save(model.state_dict(), ckpt_dir / f"{name}.pt")

        cm = confusion_matrix(test_metrics["true"], test_metrics["pred"], labels=list(range(10)))
        np.save(out_dir / f"{name}_confusion.npy", cm)

    with open(out_dir / "experiments.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    pd.DataFrame(rows).sort_values("test_acc", ascending=False).to_csv(out_dir / "summary.csv", index=False)

if __name__ == "__main__":
    main()
