# Discussion — Why WCED Pretraining Helps

The ablation result rules out the possibility that the V7b architecture alone (depth, width,
Genomic RoPE) explains its downstream performance: holding architecture fixed and varying only
the encoder's initialization produces a large, consistent gap in favor of WCED pretraining. This
is a substantially larger effect than an earlier pretrained-vs-scratch comparison found on a
smaller, non-RoPE architecture, suggesting the benefit of WCED pretraining grows with model
capacity — plausibly because a larger encoder has more capacity to encode a useful
representation during pretraining, and correspondingly more to learn from scratch when it is
not pretrained. We consider several mechanisms that plausibly explain why WCED pretraining
produces a representation this useful.

**A harsher reconstruction bottleneck than standard masked-language modeling.** Unlike
BERT-style MLM, WCED does not replace held-out CpGs with a mask-token placeholder — the unseen
CpGs are absent from the input sequence entirely. The decoder must reconstruct the full CpG
profile from a single pooled ([CLS]) vector, with no per-position token available for the
withheld half of the input. This is a harder bottleneck than token-level MLM, where a masked
position retains a placeholder embedding and can often be predicted from nearby unmasked
context. Forcing all reconstruction through one vector is what forces that vector to summarize
the sample globally, rather than encode local, position-specific features.

**The contrastive objective prevents this bottleneck from degenerating.** A reconstruction-only
objective can still collapse: the decoder could learn to output close to the population-average
methylation profile without the pooled vector carrying sample-specific information. We tested
this directly on the trained V7b pretrain checkpoint with a two-view consistency check (n=2,000
samples): the two independently-masked views of the same sample produce CLS embeddings with mean
cosine similarity 0.983, while CLS embeddings of different samples average only 0.500 — an
alignment gap of 0.484. Nearest-neighbor retrieval of each view's true matching partner succeeds
73.5% of the time out of 2,000 candidates, far above the ~0.05% expected by chance. This indicates
the encoder maps same-sample views to nearly the same representation while keeping different
samples well separated, consistent with the contrastive objective successfully preventing
representational collapse. The contrastive (InfoNCE) term is the mechanism responsible: because
the two augmented views are built from independently-sampled random subsets of the same sample's
CpGs, the only signal that can make their CLS embeddings agree is sample identity — information
about which particular CpGs happened to be visible, or population-level
statistics, cannot satisfy this constraint.

**Bidirectional attention, unlike causal/autoregressive pretraining.** DNA methylation array
data has no inherent sequential or causal order among CpGs. WCED's bidirectional attention
allows every CpG to condition on every other CpG; an autoregressive objective, by contrast, would
restrict each position to only the CpGs preceding it in an arbitrary tokenization order, which
has no biological justification for this data type.

**Genomic RoPE encodes biologically meaningful position information.** Position identifiers are
assigned by chromosomal rank rather than arbitrary vocabulary order, allowing relative-position
attention to reflect genomic proximity — informative because nearby CpGs are frequently
co-regulated. We confirmed directly that this is reflected in the trained representations, not
just the design intent: measuring cosine similarity between contextualized (post-transformer)
CpG embeddings as a function of genomic-rank distance, adjacent CpGs (distance = 1) have mean
similarity 0.39, decaying toward a random-pair baseline of 0.208 by a genomic-rank distance of
roughly 100. The *raw*, non-contextualized token embedding table shows no such relationship
(~0 similarity regardless of distance) — indicating the locality structure is imposed by the
transformer's use of Genomic RoPE during the forward pass, not already present in the embedding
table itself. Attention-selectivity analysis of the trained encoder shows this structure is used
non-trivially: certain early-layer heads attend sharply to a small, consistent subset of CpGs
(the top 1% most-attended CpGs account for the large majority of that head's attention mass),
while deeper layers integrate information more broadly — a pattern consistent with structured,
differentiated computation rather than uniform averaging across the input.

**Pretraining exploits a much larger unlabeled corpus than any single fine-tuning fold's labeled
data.** The model is pretrained on an order of magnitude more samples than are available as
labeled training data in any individual fine-tuning fold, while fitting several million encoder
parameters plus a large CpG embedding table. Learning both a general representation and the
age-prediction function simultaneously from the smaller labeled set, as the random-init model
must, is a substantially harder estimation problem than fine-tuning a lightweight head on top of
an already well-organized representation — the standard rationale for self-supervised
pretraining, for which this ablation provides direct empirical support in this setting.

## Why the benefit transfers specifically to age prediction

Probing the WCED-pretrained representation before any age supervision shows that it is already
substantially organized by age: a simple linear probe on the pretrained [CLS] embedding achieves
a much higher age R² than chance, alongside strong tissue-type and dataset discriminability. This
is consistent with the fact that age-related methylation drift is a well-documented, genome-wide
axis of variation in DNA methylation data [CITE: epigenetic-clock literature, e.g. Horvath 2013],
comparable in scale to tissue identity. A representation trained to be a faithful, sample-specific
summary of the whole methylome — which is what the reconstruction-plus-contrastive objective
enforces — should be expected to already capture much of this variation as a byproduct, without
ever being told what age is. Fine-tuning then mostly needs to *read out* an axis of variation that
is already present and well-separated in the representation, consistent with the pretrained
model's fast convergence (within roughly 20 epochs) relative to the random-init model. Fine-tuning
also further sharpens this structure specifically along the task-relevant axis: representation
geometry measured before and after fine-tuning shows increased effective rank and decreased
anisotropy in the [CLS] embedding space, while tissue- and dataset-discriminability are
essentially preserved — indicating fine-tuning specializes the representation for age prediction
without destroying the broader biological structure already present from pretraining.

## Summary

These results are consistent with WCED pretraining producing a useful representation because its
training objective is deliberately harder and more global than standard masked-language
modeling — reconstructing an entire CpG profile from one pooled vector, under a contrastive
constraint that requires that vector to encode genuine sample identity rather than population
statistics. Because age is one of the dominant axes of variation in DNA methylation, a
representation optimized to summarize a sample faithfully ends up substantially organized by age
before ever seeing a label, which plausibly explains both the fast fine-tuning convergence
observed from WCED-pretrained checkpoints and the large performance gap relative to training the
same architecture from scratch.
