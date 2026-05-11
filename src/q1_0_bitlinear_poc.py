"""Q1_0_g128 BitLinear — minimal proof-of-concept for native QAT in the
Bonsai / llama.cpp deployed 1-bit format.

Format recap:
- Group 128 consecutive input-dim positions into a "block."
- Each block has 1 sign bit per weight + 1 FP16 scale.
- Deployed value: `w_i = sign_i * scale_g` where
  `scale_g = mean(|w|)` over the 128-block.

This module:
- Maintains an FP32 shadow weight as the trainable parameter.
- In forward, derives (sign, scale) from the shadow and applies the
  quantised weight to the input. STE in backward (gradient flows
  through quantiser unchanged).
- Provides `export_q1_0()` to extract the packed bytes a GGUF
  Q1_0_g128 tensor would carry.

Why STE with a shadow is what you want (and not "train scale
directly"):
Under STE the trainable shadow's |.| changes shift the per-block
scale automatically (since scale = mean(|shadow|)), and sign flips
happen as discrete zero-crossings of shadow values. So "scale moves
continuously, signs flip rarely" is the natural dynamic — no
special parameterisation needed.

Run: `python src/q1_0_bitlinear_poc.py`
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_q1_0_g128(weight: torch.Tensor, group_size: int = 128):
    """Return (sign in {-1,+1}, per-block scale)."""
    out, in_ = weight.shape
    assert in_ % group_size == 0, f"in_features={in_} not divisible by {group_size}"
    blocks = weight.view(out, in_ // group_size, group_size)
    scale = blocks.abs().mean(dim=-1)               # (out, n_blk)
    sign = blocks.sign()
    sign = sign + (sign == 0).float()               # map 0 -> +1
    return sign.view(out, in_), scale


class Q1_0_g128_Linear(nn.Module):
    """Linear whose deployed weight is Q1_0_g128. Shadow trains via STE."""

    def __init__(self, in_features: int, out_features: int,
                 group_size: int = 128, bias: bool = False):
        super().__init__()
        assert in_features % group_size == 0
        self.in_features, self.out_features = in_features, out_features
        self.group_size = group_size
        self.weight_shadow = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight_shadow, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(cls, lin: nn.Linear, group_size: int = 128) -> "Q1_0_g128_Linear":
        m = cls(lin.in_features, lin.out_features, group_size=group_size,
                bias=lin.bias is not None)
        with torch.no_grad():
            m.weight_shadow.copy_(lin.weight)
            if lin.bias is not None:
                m.bias.copy_(lin.bias)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight_shadow
        sign, scale = quantize_q1_0_g128(W, self.group_size)
        blocks = sign.view(self.out_features, -1, self.group_size)
        W_q = (blocks * scale.unsqueeze(-1)).view(self.out_features, self.in_features)
        W_ste = W + (W_q - W).detach()              # STE
        return F.linear(x, W_ste, self.bias)

    def export_q1_0(self) -> dict[str, torch.Tensor]:
        """Pack to the bytes a Q1_0 GGUF tensor carries: 16-byte qs + fp16 scale per block."""
        with torch.no_grad():
            sign, scale = quantize_q1_0_g128(self.weight_shadow, self.group_size)
        bits = (sign > 0).to(torch.uint8)
        out, in_ = bits.shape
        bits = bits.view(out, in_ // 8, 8)          # 8 bits per byte
        powers = (1 << torch.arange(8, dtype=torch.uint8, device=bits.device))
        qs = (bits * powers).sum(dim=-1).to(torch.uint8)  # LSB-first within byte
        return {"scale": scale.to(torch.float16), "qs": qs}


# ---------------------------------------------------------------------------

def _smoke_test():
    torch.manual_seed(0)
    in_dim, hid, out_dim = 256, 512, 10
    x = torch.randn(1024, in_dim)
    y = torch.randint(0, out_dim, (1024,))

    fp = nn.Sequential(nn.Linear(in_dim, hid, bias=False), nn.ReLU(),
                       nn.Linear(hid, out_dim, bias=False))
    bit = nn.Sequential(Q1_0_g128_Linear.from_linear(fp[0]), nn.ReLU(),
                        Q1_0_g128_Linear.from_linear(fp[2]))

    opt = torch.optim.AdamW(bit.parameters(), lr=1e-3)
    print(f"{'step':>4}  {'loss':>7}  {'sign-match-vs-init':>20}")
    init_sign = torch.sign(bit[0].weight_shadow.detach().clone())
    for step in range(200):
        loss = F.cross_entropy(bit(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0 or step == 199:
            cur_sign = torch.sign(bit[0].weight_shadow.detach())
            match = (cur_sign == init_sign).float().mean().item()
            print(f"{step:>4}  {loss.item():>7.4f}  {match:>20.4f}")

    # Verify export roundtrip
    layer = bit[0]
    ex = layer.export_q1_0()
    # unpack qs -> {-1,+1} signs, compare against expected
    unpacked = ((ex["qs"].unsqueeze(-1) >>
                 torch.arange(8, dtype=torch.uint8, device=ex["qs"].device)) & 1).bool()
    unpacked = (unpacked.float() * 2 - 1).view(layer.out_features, layer.in_features)
    expected_sign, expected_scale = quantize_q1_0_g128(layer.weight_shadow)
    assert torch.allclose(unpacked, expected_sign), "sign roundtrip mismatch"
    assert torch.allclose(ex["scale"].float(), expected_scale.float(), atol=1e-3), \
        "scale roundtrip mismatch"
    print("\nExport roundtrip OK.")
    print(f"  scale: shape={tuple(ex['scale'].shape)} dtype={ex['scale'].dtype}")
    print(f"  qs:    shape={tuple(ex['qs'].shape)} dtype={ex['qs'].dtype}  "
          f"(8 sign bits/byte, LSB-first)")


if __name__ == "__main__":
    _smoke_test()
