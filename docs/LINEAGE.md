# Experiment Lineage

This repository is a curated public subset of a larger private program (v13 through v41). I include only Acts I through IV: experiments that completed with log-verified outcomes.

## Public version map

| Version | Focus | In this repository | Artifact |
|---------|-------|:------------------:|----------|
| v13 | E4 factorized emb; k=96/64 | Yes | notebook `01`, act1 JSON |
| v14 | k=128/64 reserve dimension | Yes | notebook `02` |
| v15 | k=96/48 minimum k_out | Yes | notebook `03` |
| v16 | k=96/96 shape-match | Yes | notebook `04` |
| v17 | k=80/80 smaller budget | No | private |
| v18-v25 | MoS / MTP variants | No | private |
| v26-v30 | AdamW to Muon; v30 falsified | No | private |
| v31 | Per-row SparseMuon | Yes | notebook `05`, `v31_sparse_muon_train.py` |
| v32 | Zipf diagnostics + vMF head | Yes | notebooks `06`-`07`, `v32_zipf_diagnostics.py` |
| v33-v36 | Ablations, tiered head, phase gate | No | private / failed gates |
| v37 | GGRA rare attention | No | incomplete; local only |
| v38-v40 | Routed CE, contrastive, reweight | No | private |
| v41 | vMF head-only A/B vs softmax | Yes | `v41_vmf_concentrated.py`, `v32_standard_softmax_comparison.py` |

## Rationale for exclusions

- **Duplicates:** I archive only the best executed copy per version line.
- **Failed gates:** Optimizers and ablations that did not pass pre-registered criteria remain private.
- **Incomplete runs:** Interrupted experiments are not published.
- **Size:** Colab notebooks with multi-megabyte stdout are poor review surfaces; JSON summaries and scripts serve that role.
- **Supplementary threads:** One-law scaling, bounded-law analysis, and Zipf v2 manipulation reside under `archive/internal/` (gitignored).

## Narrative (four acts)

1. **Embedding geometry:** k_out dominates k_in (11.2×); shape-match at k_in=k_out (46.46 test PPL @ epoch 2).
2. **Optimizer stack:** Muon + SparseMuon (48.93 test PPL @ epoch 2).
3. **vMF vs softmax:** Head-only matched comparison (−7.4% val @ epoch 2; mechanism measured); H1 weak, H4 confirmed.
4. **Scaling:** Chinchilla-E confirmed on OWT (E ≈ 2.49 nats); log-only replication on Pythia/Pile (E ≈ 1.29) and Meta Step-2 (E ≈ 1.65) with zero training; β_rep ∝ √N failed; V²/T not supported (with protocol audit).

## Related work

Instruction-driven code editing, difference-native pretraining, and a Chinchilla slope-one meta-analysis (held during paper review) are separate projects and are not linked from this repository.
