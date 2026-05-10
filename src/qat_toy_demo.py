# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jonah Yolles-Murphy (TG-Techie)
"""End-to-end QAT toy demo: train a tiny char-level transformer whose every
matrix-heavy weight lives on a Bonsai-style ±s_g binary lattice.

This is the smallest possible thing that lets us say, with grounded
evidence and not just inferred-from-outputs, that:

  1. A BitLinear-style straight-through-estimator QAT actually trains.
  2. Trained weights end up exactly on {±s_g}, just like Bonsai-unpacked.
  3. Per-block scales `s_g` end up at mean(|x|) of the trained shadow
     weights, just like ggml-quants.c reference Q1_0.
  4. The resulting model still generates reasonable text for the
     training distribution (Tiny Shakespeare).

This is *not* a recreation of Bonsai. It's a controlled instance of the
same recipe, at a scale that fits a CPU and a few minutes of wall time.

Run:
    uv run python src/qat_toy_demo.py
"""

from __future__ import annotations

import math
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# -- Q1_0_g128-style BitLinear -----------------------------------------------

class BitLinearSTE(torch.autograd.Function):
    """Forward: y = (sign(w) * mean(|w| per block)) @ x. Backward: identity."""

    @staticmethod
    def forward(ctx, w: torch.Tensor, group: int) -> torch.Tensor:
        # Reshape last dim into groups
        out_features, in_features = w.shape
        if in_features % group != 0:
            # Pad up to next multiple of `group` with zero columns; record so
            # we can crop. Keeps the demo flexible for any in_features.
            pad = group - (in_features % group)
            w = F.pad(w, (0, pad))
        groups = w.view(out_features, -1, group)
        scale = groups.abs().mean(dim=-1, keepdim=True)         # (out, n_groups, 1)
        # Mirror FP16 round-trip in the storage path (Bonsai's d is FP16).
        scale = scale.to(torch.float16).to(torch.float32)
        signs = torch.where(groups >= 0, torch.tensor(1.0), torch.tensor(-1.0))
        wq = (signs * scale).view(out_features, -1)
        if in_features % group != 0:
            wq = wq[:, :in_features]
        return wq

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # Straight-through estimator: pass gradient through unchanged.
        return grad_out, None


class BitLinear(nn.Module):
    """Linear layer whose effective weight is its Q1_0_g128 quantization."""

    def __init__(self, in_features: int, out_features: int, group: int = 128, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group = group
        # Shadow FP32 parameter — what gradients update
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wq = BitLinearSTE.apply(self.weight, self.group)
        out = x @ wq.T
        if self.bias is not None:
            out = out + self.bias
        return out

    @torch.no_grad()
    def quantized_weight(self) -> torch.Tensor:
        return BitLinearSTE.apply(self.weight, self.group)


# -- Tiny transformer --------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, group: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = BitLinear(dim, 3 * dim, group=group)
        self.proj = BitLinear(dim, dim, group=group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.ones(T, T, device=x.device).triu(1).bool()
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class FFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int, group: int):
        super().__init__()
        hidden = mlp_ratio * dim
        self.up = BitLinear(dim, hidden, group=group)
        self.down = BitLinear(hidden, dim, group=group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: int, group: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, group)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mlp_ratio, group)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyBitTransformer(nn.Module):
    def __init__(self, vocab: int, dim: int = 128, n_heads: int = 4, n_layers: int = 4,
                 mlp_ratio: int = 4, max_seq: int = 256, group: int = 128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(max_seq, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads, mlp_ratio, group) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = BitLinear(dim, vocab, group=group)
        self.max_seq = max_seq

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, n: int = 200, temperature: float = 0.8, top_k: int = 50):
        for _ in range(n):
            ctx = idx[:, -self.max_seq:]
            logits = self(ctx)[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


# -- Driver -------------------------------------------------------------------

def fetch_corpus(path: Path) -> str:
    """Tiny Shakespeare (1.1 MB) from karpathy/char-rnn."""
    if not path.exists():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    return path.read_text()


def build_vocab(text: str) -> tuple[dict, dict]:
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def decode(t: torch.Tensor, itos: dict) -> str:
    return "".join(itos[int(i)] for i in t)


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, data.numel() - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in idx])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in idx])
    return x, y


def main():
    REPO = Path(__file__).resolve().parent.parent
    DATA_DIR = REPO / "models" / "qat_demo"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    text = fetch_corpus(DATA_DIR / "tinyshakespeare.txt")
    stoi, itos = build_vocab(text)
    vocab = len(stoi)
    print(f"vocab={vocab}  corpus={len(text)} chars")

    torch.manual_seed(0)

    # Hyperparameters tuned for ~5 min CPU run.
    DIM = 128
    N_HEADS = 4
    N_LAYERS = 4
    MLP_RATIO = 4
    GROUP = 128            # match Bonsai's Q1_0_g128
    SEQ_LEN = 128
    BATCH = 16
    STEPS = 800
    LR = 3e-3
    LOG_EVERY = 50

    data = encode(text, stoi)
    n_train = int(0.9 * data.numel())
    train_data = data[:n_train]
    val_data = data[n_train:]

    model = TinyBitTransformer(vocab, dim=DIM, n_heads=N_HEADS, n_layers=N_LAYERS,
                               mlp_ratio=MLP_RATIO, max_seq=SEQ_LEN, group=GROUP)
    n_params = sum(p.numel() for p in model.parameters())
    bit_params = sum(m.weight.numel() for m in model.modules() if isinstance(m, BitLinear))
    print(f"model: {n_params/1e3:.1f}K total params, {bit_params/1e3:.1f}K of them on the binary lattice")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    history = []

    t0 = time.time()
    for step in range(1, STEPS + 1):
        model.train()
        x, y = get_batch(train_data, BATCH, SEQ_LEN)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % LOG_EVERY == 0 or step == 1:
            with torch.no_grad():
                model.eval()
                vx, vy = get_batch(val_data, BATCH, SEQ_LEN)
                vloss = F.cross_entropy(model(vx).view(-1, vocab), vy.view(-1)).item()
            history.append({"step": step, "train_loss": loss.item(), "val_loss": vloss,
                            "elapsed_s": time.time() - t0})
            print(f"step {step:4d}  train={loss.item():.3f}  val={vloss:.3f}  "
                  f"elapsed={history[-1]['elapsed_s']:.1f}s")

    # ---- empirical checks on the trained model -------------------------------
    print("\n== H1-style check on the trained QAT model ==")
    print("  for each BitLinear, the effective forward weight should live on ±s_g")
    print("  (i.e. for each 128-block, all magnitudes equal a single FP16 scale).")
    for name, mod in model.named_modules():
        if isinstance(mod, BitLinear):
            wq = mod.quantized_weight()
            grouped = wq.view(mod.out_features, -1, GROUP)
            abs_g = grouped.abs()
            # Round to FP16 ULP to absorb cast noise
            rounded = (abs_g * 1e4).round() / 1e4
            distinct = torch.tensor([torch.unique(row).numel() for row in rounded.view(-1, GROUP)])
            frac_binary = float((distinct == 1).float().mean())
            print(f"    {name:30s}  shape={tuple(wq.shape)}  frac_binary_lattice={frac_binary:.4f}")

    print("\n== sample generation ==")
    model.eval()
    seed = encode("ROMEO:", stoi).unsqueeze(0)
    out = model.generate(seed, n=200)
    print(decode(out[0], itos))

    out_path = REPO / "reports" / "qat_demo" / "training_log.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"vocab={vocab}  corpus={len(text)} chars\n")
        f.write(f"model: {n_params/1e3:.1f}K params, {bit_params/1e3:.1f}K on binary lattice\n")
        for h in history:
            f.write(f"step {h['step']:4d}  train={h['train_loss']:.3f}  val={h['val_loss']:.3f}  "
                    f"elapsed={h['elapsed_s']:.1f}s\n")
        f.write("\n--- sample (seed='ROMEO:') ---\n")
        f.write(decode(out[0], itos))
    print(f"\n[saved] {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
