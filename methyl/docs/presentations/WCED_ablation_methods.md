# Methods — WCED Pretraining Ablation

To isolate the contribution of WCED pretraining from the contribution of the MethylLlama V7b
architecture itself, we trained two models that are identical in every respect except encoder
initialization:

- **WCED-pretrained**: encoder initialized from the WCED pretraining checkpoint (6 transformer
  layers, 256-dimensional hidden state, 4 attention heads, Genomic RoPE positional encoding),
  then fine-tuned for age prediction.
- **Random-init**: an encoder of identical architecture (same layer count, hidden size, head
  count, and Genomic RoPE configuration), initialized randomly and fine-tuned from scratch, with
  no pretrained weights.

Both models used the same fold-0 train/validation/test split of the 21k AltumAge dataset, the
same fine-tuning head, the same loss function, the same learning-rate schedule, and the same
per-epoch training budget. The random-init model additionally left the CpG embedding table and
encoder unfrozen from the first epoch (rather than frozen-then-unfrozen at epoch 10, as used for
the pretrained model), since there were no pretrained weights to protect during an initial
warm-up phase. All other hyperparameters were held identical between the two runs.

Both models were evaluated at a matched epoch (epoch 148, the latest epoch common to both
training runs within the analysis window) and, separately, on the fixed held-out test set using
each run's best validation-MedAE checkpoint.
