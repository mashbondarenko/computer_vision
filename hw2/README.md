# HW2: Vision Transformer (ViT) for Digit Classification

Minimal Vision Transformer study on the sklearn Digits dataset (8×8 grayscale, 10 classes).

## Variants

| Variant | Patch | Depth | Head type | Test acc |
|---------|-------|-------|-----------|----------|
| A | 2 | 2 | cls token | 92.8% |
| B | 4 | 2 | cls token | 86.9% |
| C | 2 | 4 | cls token | 93.6% |
| D | 2 | 2 | mean pool | 93.1% |

## Files

- `src/` — model, data loading, training, figure generation
- `checkpoints/` — saved model weights (.pt)
- `results/` — confusion matrices, metrics
- `figures/` — training curves, attention maps, confusion matrices
- `report.md` — full experimental report

## Run

```bash
pip install -r requirements.txt
python src/run_experiments.py
```
