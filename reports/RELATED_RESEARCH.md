# Related research — prior art that bears on Bonsai's technique

> Derived by two worktree-isolated, fresh-context sub-agents. The first was tasked with broad 1-bit / sub-2-bit literature against the empirical fingerprint we'd reconstructed; the second was narrowed to the people behind PrismML and their own publication trail. Outputs are summarised below; both agents' independence keeps the framing honest. URLs are arXiv stable links the maintainer can fetch and add to the repo.

## PrismML team (from public press; prismml.com/about returns 403)

Four named co-founders, all Caltech-rooted:

- **Babak Hassibi** — Founder + CEO. Mose & Lilian S. Bohn Professor of EE at Caltech. Information theory, signal processing, control, ML. The whitepaper's "proprietary Caltech IP" is presumably his.
- **Sahin Lale** — Co-founder, Co-Head of Research. Caltech PhD; reinforcement-learning / adaptive-control under Hassibi & Anandkumar. No published work on quantisation specifically.
- **Omead Pooladzandi** — Co-founder, Co-Head of Research. UCLA PhD ("Fast Training of Generalizable Deep Neural Networks", 2023). Closest publication to compression: VQ-GAN bottlenecks for poison defence (arXiv:2509.25792) — tangential.
- **Reza Sadri** — Co-founder, VP Strategy. Director of Caltech AI Bootcamp; ex-Instacart ML infra. No academic compression record.

**Whitepaper bibliographies cite no PrismML-authored prior work** — only third-party benchmarks (BitNet, Qwen3, GGUF, MLX, llama.cpp, etc.). The "Caltech IP" is deliberately not bibliographically anchored.

## Hassibi-group prior art (highest value)

Listed most-relevant first.

1. **Akhtiamov, Ghane, Hassibi — "One-Bit Quantization and Sparsification for Multiclass Linear Classification with Strong Regularization"** (Feb 2024, rev. Oct 2024). [arXiv:2402.10474](https://arxiv.org/abs/2402.10474). Proves that ℓ∞ regularisation concentrates weights on two opposite-sign extreme values, allowing 1-bit compression with minimal performance loss. **Predicts both** the ±s_g binary lattice (force-by-format) **and** the inflated-beyond-RMSE-optimal magnitudes we observe (~2× ratio). The theoretical anchor most consistent with "Caltech IP".
2. **Akhtiamov, Ghane, Hassibi — "One-Bit Quantization for Random Features Models"** (Oct 2025). [arXiv:2510.16250](https://arxiv.org/abs/2510.16250). Proves that 1-bit quantising **every layer except the last** is asymptotically lossless in generalisation for the random-features model. **Predicts** observation 7: lm_head treated specially (separate tensor at 8B with scales recomputed, signs largely preserved).
3. **Hassibi & Stork — "Optimal Brain Surgeon and General Network Pruning"** (1992/1993). [Caltech archive](https://authors.library.caltech.edu/records/tzehb-vz706). Foundational second-order Hessian-based pruning. The lineage GPTQ/SparseGPT/BiLLM/OBC all derive from. Hassibi's own original work — the obvious starting point if his group were extending it to 1-bit.
4. **Akhtiamov, Bosch, Ghane, Varma, Hassibi — "A Novel Gaussian Min-Max Theorem and its Applications"** (Feb 2024). [arXiv:2402.07356](https://arxiv.org/abs/2402.07356). CGMT machinery underpinning the high-dim error analyses used in (1)(2).
5. **Akhtiamov, Ghane, Hassibi — "A Precise Performance Analysis of the Randomized Singular Value Decomposition"** (Oct 2025). [arXiv:2510.06490](https://arxiv.org/abs/2510.06490). Cousin compression-theory technique.
6. **Kostina, Hassibi — "Rate-Cost Tradeoffs in Control"** (Dec 2016). [arXiv:1612.02126](https://arxiv.org/abs/1612.02126). Closest pure rate-distortion paper Hassibi has — the framing the whitepaper invokes for "intelligence density".

## Closely-related public 1-bit / sub-2-bit techniques

Not Hassibi-authored; included because they could match aspects of our fingerprint.

7. **Huang et al. — BiLLM** (ICML 2024). [arXiv:2402.04291](https://arxiv.org/abs/2402.04291). Hessian-guided salient-vs-non-salient split, residual binarisation for salient weights, optimal-splitting-search for non-salient blocks. **Strongly explains** observations 4 (per-block scales ≠ `mean(|w|)`, they come from a layerwise optimal-split search) and 8 (erratic depth behaviour because per-layer salience masks differ).
8. **Xu et al. — OneBit** (NeurIPS 2024). [arXiv:2402.11295](https://arxiv.org/abs/2402.11295). SVID factorisation `W = Sign(W) · diag(a) · diag(b)`, QAT with KD from base. **Explains** signs anchored to base + scales recomputed from FP factors + KD step as natural injection point for identity data.
9. **Zhao et al. — PTQ1.61** (Feb 2025). [arXiv:2502.13179](https://arxiv.org/abs/2502.13179). Block-wise scaling factors optimised separately from `mean(|w|)`, 1D structured salience mask, "quantization preprocessing" weight-distribution transform. **Explains observation 4 directly.**
10. **Hoang et al. — "Rethinking Output Alignment for 1-bit PTQ"** (Dec 2025). [arXiv:2512.21651](https://arxiv.org/abs/2512.21651). Identifies cross-layer error accumulation; selective layerwise output alignment + attention-aware masking. **Predicts** the erratic depth-r² + the input_layernorm-modified-to-absorb-forward-error pattern (peak at L0, decay toward output).
11. **Yang et al. — "Rethinking 1-bit Optimization Leveraging Pre-trained LLMs"** (Aug 2025). [arXiv:2508.06974](https://arxiv.org/abs/2508.06974). Progressive forward+backward binarisation, binary-aware initialisation, dual-scaling compensation. **Could explain** observation 10: bigger Bonsai stays closer to base (more capacity headroom for progressive training).
12. **Dong et al. — STBLLM** (ICLR 2025). [arXiv:2408.01803](https://arxiv.org/abs/2408.01803). Per-layer N:M sparsity ratios. **Explains** non-monotonic per-layer compression aggressiveness.
13. **Frantar, Ashkboos, Hoefler, Alistarh — GPTQ** (ICLR 2023). [arXiv:2210.17323](https://arxiv.org/abs/2210.17323). Sequential layerwise OBS-derived quantisation — Hassibi-lineage. Doesn't natively reach 1-bit, but its inverse-Hessian update would explain why scales drift from `mean(|w|)`.
14. **Sean I. Young — Radio** (ICML 2025). [arXiv:2505.03031](https://arxiv.org/abs/2505.03031). Rate-distortion-optimised LLM quantisation. Mirrors the whitepaper's "rate-distortion / intelligence density" framing exactly. (Young is *not* PrismML; just topical.)

## Synthesis (not a confirmation, just a directional read)

The combination most consistent with our empirical fingerprint:

- **Theoretical anchor: Hassibi's ℓ∞-regularisation 1-bit result (2402.10474)** — explains why the binary lattice with inflated magnitudes is the *fixed point*, not just a useful approximation.
- **Optimisation procedure resembling OneBit (2402.11295) + BiLLM/PTQ1.61 block-scale search** — sign-anchored decomposition initialised from base, with per-block scales chosen by an optimisation that doesn't reduce to `mean(|w|)`.
- **Forward-output-alignment pass (Hoang 2512.21651)** — propagates per-layer error into input_layernorm; explains the layer-0 peak and L35 untouchedness in our norm-depth scan.
- **A KD or instruction-tune step** — supplies the "Bonsai/PrismML/Caltech" identity, separately from the quantisation.
- The **last layer (lm_head)** treated separately, as Hassibi's 2510.16250 result motivates.

This is one possible reading. Other combinations are still consistent with the data; see `HYPOTHESIS_SPACE.md` for H_a/H_b/H_c/H_d.

## Suggested PDFs to add to the repo

The maintainer can fetch and commit to `reports/related_papers/` or similar. The Hassibi-authored ones are highest priority:

- https://arxiv.org/abs/2510.16250 — Hassibi 1-bit RF (last-layer-special)
- https://arxiv.org/abs/2402.10474 — Hassibi ℓ∞ 1-bit (theoretical anchor)
- https://arxiv.org/abs/2402.07356 — Hassibi CGMT machinery
- https://arxiv.org/abs/2510.06490 — Hassibi precise SVD
- https://arxiv.org/abs/1612.02126 — Hassibi rate-cost
- https://arxiv.org/abs/2402.04291 — BiLLM
- https://arxiv.org/abs/2402.11295 — OneBit
- https://arxiv.org/abs/2502.13179 — PTQ1.61
- https://arxiv.org/abs/2512.21651 — Output alignment for 1-bit PTQ
- https://arxiv.org/abs/2508.06974 — Progressive 1-bit
- https://arxiv.org/abs/2408.01803 — STBLLM
- https://arxiv.org/abs/2210.17323 — GPTQ
- https://arxiv.org/abs/2505.03031 — Radio rate-distortion
- https://authors.library.caltech.edu/records/tzehb-vz706 — Optimal Brain Surgeon (1992)
