# TextCNN on MELD – experiment results

All runs use the official MELD splits (train 9989 / dev 1109 / test 2610 utterances),
vocabulary built from train only, best checkpoint selected on dev macro F1, and
metrics reported on test. Each folder holds:

- `training_curves.png` – loss, accuracy and macro F1 per epoch (train vs dev), best epoch marked
- `confusion_matrix_test.png` – test confusion matrix over the 7 emotions (counts + row-normalized)
- `history.csv` – per-epoch train/dev metrics

Folders are cumulative: each one keeps the changes of the previous ones unless stated otherwise.

| Folder | Change | Script |
|---|---|---|
| `01_class_weights_early_stopping` | Embeddings trained from scratch, inverse-frequency class-weighted loss, early stopping (patience 3 on dev macro F1), seed 42 | `python textcnn.py` |
| `02_glove_frozen` | GloVe 6B 100d embeddings, frozen | `python textcnn_glove.py` |
| `03_glove_finetuned` | GloVe embeddings fine-tuned (same LR as the rest, 1e-3) | `FREEZE_EMBEDDINGS=0 python textcnn_glove.py` |
| `04_glove_finetuned_sqrt_weights` | Square-root class weights instead of inverse-frequency (3 seeds) | `FREEZE_EMBEDDINGS=0 CLASS_WEIGHTING=sqrt SEED=<s> python textcnn_glove.py` |
| `05_glove_finetuned_sqrt_weights_emb_lr1e-4_wd0.01` | Embedding LR 1e-4 + AdamW weight decay 0.01 on non-embedding layers (3 seeds) | `FREEZE_EMBEDDINGS=0 CLASS_WEIGHTING=sqrt EMBEDDING_LR=1e-4 WEIGHT_DECAY=1e-2 SEED=<s> python textcnn_glove.py` |

GloVe vectors are expected at `../../embeddings/glove.6B.100d.txt` (outside the repo).

## Test results

Single seed (42):

| Run | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| 01 class weights + early stopping | 0.352 | 0.242 | 0.359 |
| 02 GloVe frozen | 0.348 | 0.240 | 0.357 |
| 03 GloVe fine-tuned | 0.348 | 0.266 | 0.352 |

Mean ± std over seeds 42, 1, 2:

| Run | Accuracy | Macro F1 | Weighted F1 | Test loss |
|---|---|---|---|---|
| 04 GloVe fine-tuned + sqrt weights | 0.476 ± 0.009 | **0.297 ± 0.006** | 0.456 ± 0.004 | 2.21 |
| 05 + embedding LR 1e-4, weight decay 0.01 | **0.487 ± 0.004** | 0.295 ± 0.020 | **0.462 ± 0.009** | **1.91** |

Per seed (accuracy / macro F1 / weighted F1):

| Run | seed 42 | seed 1 | seed 2 |
|---|---|---|---|
| 04 | 0.470 / 0.304 / 0.454 | 0.472 / 0.295 / 0.453 | 0.487 / 0.292 / 0.460 |
| 05 | 0.487 / 0.314 / 0.469 | 0.490 / 0.298 / 0.464 | 0.483 / 0.275 / 0.452 |

Reference: BiLSTM (`biltsm/results_run.txt`) – accuracy 0.588, macro F1 0.384, weighted F1 0.571.

## Takeaways

- GloVe fine-tuning and square-root class weights are the changes that helped most
  (macro F1 0.24 → ~0.30, accuracy back to ~0.48 after inverse weights had dropped it to ~0.35).
- Lower embedding LR + weight decay does not improve macro F1 over 3 seeds (0.295 vs 0.297);
  it gives slightly higher accuracy / weighted F1 and lower test loss (less overconfident),
  but with higher seed-to-seed variance.
- Overfitting remains in all runs (train macro F1 keeps rising while dev plateaus around 0.28–0.32).
