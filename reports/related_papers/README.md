# Related papers — local cache

A directory of PDFs that bear on Bonsai's reverse-engineering and on candidate
techniques. Listed in `reports/RELATED_RESEARCH.md` with one-line summaries
and which empirical observation each does or doesn't predict.

## Conventions for adding more papers

- File name format: `<Short Name>- <Title> - <arXiv ID>v<N>.pdf` (matches the
  initial drop). Lowercase `v` for the arXiv version suffix; the arXiv ID is
  the canonical anchor.
- Add a one-paragraph entry in `reports/RELATED_RESEARCH.md` for each new
  paper, with: arXiv URL, what the paper does in one sentence, and which
  empirical observation from `ASSUMPTIONS_AUDIT.md §1` it explains or
  refutes.
- Don't store derived analysis here. PDFs are read-only inputs.

## Currently in the cache

PrismML/Hassibi-group prior art (highest priority — these are the actual
people behind Bonsai):

- `ONE-BIT QUANTIZATION AND SPARSIFICATION FOR MULTICLASS LINEAR CLASSIFICATION VIA REGULARIZED REGRESSION - 2402.10474v2.pdf` — [arXiv 2402.10474](https://arxiv.org/abs/2402.10474). Akhtiamov–Ghane–Hassibi, Feb 2024 / rev. Oct 2024.
- `One-Bit Quantization for Random Features Models - 2510.16250v1.pdf` — [arXiv 2510.16250](https://arxiv.org/abs/2510.16250). Akhtiamov–Ghane–Hassibi, Oct 2025.

Public 1-bit / sub-2-bit techniques whose empirical fingerprints we may want
to compare against Bonsai's:

- `BiLLM- Pushing the Limit of Post-Training Quantization for LLMs - 2402.04291v2.pdf`
- `OneBit- Towards Extremely Low-bit Large Language Models - 2402.11295v6.pdf`
- `PTQ1.61- Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models - 2502.13179v2.pdf`
- `Rethinking 1-bit Optimization Leveraging Pre-trained Large Language Models - 2508.06974v1.pdf`
- `RETHINKING OUTPUT ALIGNMENT FOR 1-BIT POST- TRAINING QUANTIZATION OF LARGE LANGUAGE MODELS - 2512.21651v1.pdf`
- `STBLLM- BREAKING THE 1-BIT BARRIER WITH STRUCTURED BINARY LLMS - 2408.01803v2.pdf`
- `GPTQ- ACCURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PRE-TRAINED TRANSFORMERS - 2210.17323v2.pdf`
- `Radio- Rate–Distortion Optimization for Large Language Model Compression - 2505.03031v1.pdf`

## Open candidates (not yet downloaded)

From `reports/RELATED_RESEARCH.md`:

- Hassibi & Stork — *Optimal Brain Surgeon and General Network Pruning* (1992/1993). [Caltech archive](https://authors.library.caltech.edu/records/tzehb-vz706). Foundational Hessian-based pruning; the lineage GPTQ/SparseGPT/BiLLM/OBC all derive from. Worth grabbing.
- Akhtiamov, Bosch, Ghane, Varma, Hassibi — *A Novel Gaussian Min-Max Theorem and its Applications* (Feb 2024). [arXiv 2402.07356](https://arxiv.org/abs/2402.07356). CGMT machinery underpinning the Hassibi 2024/2025 1-bit results.
- Akhtiamov, Ghane, Hassibi — *A Precise Performance Analysis of the Randomized Singular Value Decomposition* (Oct 2025). [arXiv 2510.06490](https://arxiv.org/abs/2510.06490).
- Kostina, Hassibi — *Rate-Cost Tradeoffs in Control* (Dec 2016). [arXiv 1612.02126](https://arxiv.org/abs/1612.02126). Closest pure rate-distortion paper Hassibi has.
