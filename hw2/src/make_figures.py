import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from model import VisionTransformer

CONFIGS = {
    "A_patch2_depth2_cls": dict(image_size=8, patch_size=2, embed_dim=32, depth=2, heads=4, use_cls_token=True, dropout=0.0),
    "B_patch4_depth2_cls": dict(image_size=8, patch_size=4, embed_dim=32, depth=2, heads=4, use_cls_token=True, dropout=0.0),
    "C_patch2_depth4_cls": dict(image_size=8, patch_size=2, embed_dim=32, depth=4, heads=4, use_cls_token=True, dropout=0.0),
    "D_patch2_depth2_mean": dict(image_size=8, patch_size=2, embed_dim=32, depth=2, heads=4, use_cls_token=False, dropout=0.0),
}

def load_raw_test_split(seed: int = 42):
    X, y = load_digits(return_X_y=True)
    X = X.reshape(-1, 8, 8).astype(np.float32) / 16.0
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)
    return X_test, y_test

def make_patch_grid_image(img8, patch_size, out_path, title_text):
    arr = (img8 * 255).astype(np.uint8)
    scale = 32
    img = Image.fromarray(arr).resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.Resampling.NEAREST).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    step = patch_size * scale
    for x in range(0, w + 1, step):
        draw.line((x, 0, x, h), width=2)
    for y in range(0, h + 1, step):
        draw.line((0, y, w, y), width=2)
    canvas = Image.new("RGB", (w, h + 40), "white")
    canvas.paste(img, (0, 40))
    d = ImageDraw.Draw(canvas)
    d.text((10, 10), title_text, fill="black")
    canvas.save(out_path)

def make_error_collage(error_df, X_test_raw, title, out_path, max_items=8):
    items = error_df.head(max_items).to_dict(orient="records")
    cell_w, cell_h = 180, 180
    header_h = 40
    cols = 4
    rows = math.ceil(len(items) / cols)
    canvas = Image.new("RGB", (cols * cell_w, header_h + rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill="black")
    for k, item in enumerate(items):
        r = k // cols
        c = k % cols
        x0 = c * cell_w
        y0 = header_h + r * cell_h
        img = (X_test_raw[item["idx"]] * 255).astype(np.uint8)
        tile = Image.fromarray(img).resize((96, 96), Image.Resampling.NEAREST).convert("RGB")
        canvas.paste(tile, (x0 + 8, y0 + 8))
        text = f'idx={item["idx"]}\ntrue={item["true"]} pred={item["pred"]}\nconf={item["conf_pred"]:.2f}'
        draw.multiline_text((x0 + 110, y0 + 20), text, fill="black", spacing=4)
    canvas.save(out_path)

def make_compare_collage(df, X_test_raw, title, out_path, max_items=8):
    items = df.head(max_items).to_dict(orient="records")
    cell_w, cell_h = 220, 180
    header_h = 40
    cols = 4
    rows = math.ceil(len(items) / cols)
    canvas = Image.new("RGB", (cols * cell_w, header_h + rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill="black")
    for k, item in enumerate(items):
        r = k // cols
        c = k % cols
        x0 = c * cell_w
        y0 = header_h + r * cell_h
        img = (X_test_raw[item["idx"]] * 255).astype(np.uint8)
        tile = Image.fromarray(img).resize((96, 96), Image.Resampling.NEAREST).convert("RGB")
        canvas.paste(tile, (x0 + 8, y0 + 8))
        text = f'idx={item["idx"]}\ntrue={item["true"]}\nA->{item["pred_A"]} ({item["A_conf"]:.2f})\nB->{item["pred_B"]} ({item["B_conf"]:.2f})'
        draw.multiline_text((x0 + 110, y0 + 10), text, fill="black", spacing=4)
    canvas.save(out_path)

def probs_from_logits(logits):
    logits = np.asarray(logits)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    return probs / probs.sum(axis=1, keepdims=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.repo_root)
    results_dir = root / "results"
    ckpt_dir = root / "checkpoints"
    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)

    with open(results_dir / "experiments.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    X_test_raw, y_test = load_raw_test_split(seed=args.seed)

    # tokenization images
    sample_idx = 2
    make_patch_grid_image(X_test_raw[sample_idx], 2, fig_dir / "tokenization_patch2.png", "Tokenization example: patch size 2, 4x4 patches, seq_len = 16 + cls")
    make_patch_grid_image(X_test_raw[sample_idx], 4, fig_dir / "tokenization_patch4.png", "Tokenization example: patch size 4, 2x2 patches, seq_len = 4 + cls")

    # val curves
    plt.figure(figsize=(8, 5))
    for name in ["A_patch2_depth2_cls", "B_patch4_depth2_cls", "C_patch2_depth4_cls", "D_patch2_depth2_mean"]:
        hist = pd.DataFrame(results[name]["history"])
        plt.plot(hist["epoch"], hist["val_acc"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Validation accuracy across configurations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "val_accuracy_curves.png", dpi=180)
    plt.close()

    # confusion matrices
    for name in ["A_patch2_depth2_cls", "B_patch4_depth2_cls"]:
        cm = np.load(results_dir / f"{name}_confusion.npy") if (results_dir / f"{name}_confusion.npy").exists() else None
        if cm is None:
            continue
        plt.figure(figsize=(6, 5))
        plt.imshow(cm)
        plt.colorbar()
        plt.xticks(range(10))
        plt.yticks(range(10))
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.title(f"Confusion matrix: {name}")
        for i in range(10):
            for j in range(10):
                plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / f"confusion_{name}.png", dpi=180)
        plt.close()

    # reconstruct model outputs if checkpoints are available
    model_A = VisionTransformer(**CONFIGS["A_patch2_depth2_cls"])
    model_B = VisionTransformer(**CONFIGS["B_patch4_depth2_cls"])
    model_A.load_state_dict(torch.load(ckpt_dir / "A_patch2_depth2_cls.pt", map_location="cpu"))
    model_B.load_state_dict(torch.load(ckpt_dir / "B_patch4_depth2_cls.pt", map_location="cpu"))

    X_norm, y_full = load_digits(return_X_y=True)
    X_norm = X_norm.reshape(-1, 8, 8).astype(np.float32) / 16.0
    mean = X_norm.mean()
    std = X_norm.std() + 1e-6
    X_norm = (X_norm - mean) / std
    X_train, X_temp, y_train, y_temp = train_test_split(X_norm, y_full, test_size=0.4, random_state=args.seed, stratify=y_full)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp)

    with torch.no_grad():
        logits_A = model_A(torch.tensor(X_test[:, None], dtype=torch.float32)).numpy()
        logits_B = model_B(torch.tensor(X_test[:, None], dtype=torch.float32)).numpy()

    probs_A = probs_from_logits(logits_A)
    probs_B = probs_from_logits(logits_B)
    pred_A = logits_A.argmax(axis=1)
    pred_B = logits_B.argmax(axis=1)

    def errors_df(pred, probs):
        rows = []
        for idx, (t, p) in enumerate(zip(y_test, pred)):
            if t != p:
                rows.append({"idx": idx, "true": int(t), "pred": int(p), "conf_pred": float(probs[idx, p])})
        return pd.DataFrame(rows).sort_values("conf_pred", ascending=False)

    errors_A = errors_df(pred_A, probs_A)
    errors_B = errors_df(pred_B, probs_B)
    make_error_collage(errors_A, X_test_raw, "High-confidence errors: A_patch2_depth2_cls", fig_dir / "errors_A_patch2_depth2_cls.png")
    make_error_collage(errors_B, X_test_raw, "Errors: B_patch4_depth2_cls", fig_dir / "errors_B_patch4_depth2_cls.png")


    def attn_map_image(model, x_norm, x_raw, idx, title, out_path):
        with torch.no_grad():
            logits, attn = model(torch.tensor(x_norm[idx:idx+1, None], dtype=torch.float32), return_last_attn=True)
        attn = attn[0].mean(0).detach().cpu().numpy()
        if model.use_cls_token:
            vec = attn[0, 1:]
        else:
            vec = attn.mean(0)
        grid = model.patch_embed.grid_size if hasattr(model, "patch_embed") else model.patch.grid_size
        heat = vec.reshape(grid, grid)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((256, 256), Image.Resampling.BILINEAR)
        digit_img = Image.fromarray((x_raw[idx] * 255).astype(np.uint8)).resize((256, 256), Image.Resampling.NEAREST)
        digit_rgb = digit_img.convert("RGB")
        heat_rgb = Image.merge("RGB", (heat_img, Image.new("L", heat_img.size, 0), Image.new("L", heat_img.size, 0)))
        blended = Image.blend(digit_rgb, heat_rgb, alpha=0.35)
        canvas = Image.new("RGB", (256, 296), "white")
        canvas.paste(blended, (0, 40))
        draw = ImageDraw.Draw(canvas)
        pred = int(logits.argmax(dim=1).item())
        draw.text((10, 10), f"{title} | idx={idx}, true={int(y_test[idx])}, pred={pred}", fill="black")
        canvas.save(out_path)

    compare_rows = []
    for idx, true in enumerate(y_test):
        compare_rows.append({
            "idx": idx,
            "true": int(true),
            "pred_A": int(pred_A[idx]),
            "pred_B": int(pred_B[idx]),
            "A_correct": bool(pred_A[idx] == true),
            "B_correct": bool(pred_B[idx] == true),
            "A_conf": float(probs_A[idx, pred_A[idx]]),
            "B_conf": float(probs_B[idx, pred_B[idx]]),
        })
    compare_df = pd.DataFrame(compare_rows)
    improved_df = compare_df[(~compare_df["A_correct"]) & (compare_df["B_correct"])].sort_values("A_conf", ascending=False)
    make_compare_collage(improved_df, X_test_raw, "A wrong, B right: examples fixed by larger patches", fig_dir / "a_wrong_b_right_examples.png")

    sample_idx = 2
    attn_map_image(model_A, X_test, X_test_raw, sample_idx, "Last-layer cls attention overlay, A", fig_dir / "attention_A_sample.png")
    attn_map_image(model_B, X_test, X_test_raw, sample_idx, "Last-layer cls attention overlay, B", fig_dir / "attention_B_sample.png")

if __name__ == "__main__":
    main()
