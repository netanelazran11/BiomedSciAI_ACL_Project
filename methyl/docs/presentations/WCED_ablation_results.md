# Results — WCED Pretraining Ablation

| Metric | WCED-pretrained | Random-init | Gap |
|---|---|---|---|
| Val MedAE @ epoch 148 (matched) | 2.875 yr | 3.969 yr | 38% higher without pretraining |
| Val R² @ epoch 148 (matched) | 0.931 | 0.854 | −0.077 |
| Best val MedAE (within first 160 epochs) | 2.625 yr (ep 92) | 3.969 yr (ep 148) | 51% higher |
| Test MedAE | 3.125 yr | 4.719 yr | 51% higher |
| Test MAE | 4.440 yr | 6.749 yr | 52% higher |
| Test R² | 0.9321 | 0.8511 | −0.081 |

The WCED-pretrained model's validation MedAE plateaus by approximately epoch 20–30. The
random-init model's validation MedAE is still decreasing at epoch 160 and had not reached the
pretrained model's plateau value within that window (Figure — validation MedAE vs. epoch,
both runs). At the matched epoch 148, the pretrained model's MedAE is 38% lower than the
random-init model's; on the held-out test set, using each run's best checkpoint, the gap widens
to 51%.

For comparison, an earlier ablation of the same kind (pretrained-warmstart vs. from-scratch),
run on a smaller, non-RoPE architecture (4 layers, no Genomic RoPE), found a substantially
smaller gap: test MedAE 3.63 yr (warmstart) vs. 3.75 yr (scratch), a ~3% difference.
