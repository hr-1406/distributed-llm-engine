from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


NF4_CODEBOOK = torch.tensor(
    [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0898, 0.0,
     0.0898, 0.1848, 0.2844, 0.3949, 0.5251, 0.6962, 1.0, 1.0],
    dtype=torch.float32,
)


@dataclass
class QuantizedWeight:
    codes: torch.Tensor
    scales: torch.Tensor
    shape: torch.Size
    bits: int
    block_size: int = 1


def quantize_int8(weight: torch.Tensor) -> QuantizedWeight:
    if weight.numel() == 0:
        raise ValueError("cannot quantize an empty tensor")
    scale = weight.abs().amax().clamp_min(torch.finfo(weight.float().dtype).eps) / 127
    codes = torch.clamp(torch.round(weight.float() / scale), -127, 127).to(torch.int8)
    return QuantizedWeight(codes, scale.reshape(1), weight.shape, 8, 1)


def quantize_nf4(weight: torch.Tensor, block_size: int = 64) -> QuantizedWeight:
    if weight.numel() == 0 or block_size <= 0:
        raise ValueError("weight must be non-empty and block_size positive")
    flat = weight.float().flatten()
    padded = torch.nn.functional.pad(flat, (0, (-flat.numel()) % block_size))
    blocks = padded.reshape(-1, block_size)
    scales = blocks.abs().amax(dim=1).clamp_min(1e-8)
    normalized = (blocks / scales[:, None]).clamp(-1, 1)
    codebook = NF4_CODEBOOK.to(weight.device)
    codes = (normalized[..., None] - codebook).abs().argmin(-1).to(torch.uint8)
    return QuantizedWeight(codes.flatten()[: flat.numel()], scales, weight.shape, 4, block_size)


def dequantize(quantized: QuantizedWeight, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    if quantized.bits == 8:
        values = quantized.codes.float() * quantized.scales[0]
    elif quantized.bits == 4:
        block_size = quantized.block_size
        padded = torch.nn.functional.pad(
            quantized.codes, (0, (-quantized.codes.numel()) % block_size)
        )
        block_count = padded.numel() // block_size
        codebook = NF4_CODEBOOK.to(quantized.codes.device)
        values = (
            codebook[padded.long()].reshape(block_count, block_size) * quantized.scales[:, None]
        ).flatten()
        values = values[: quantized.codes.numel()]
    else:
        raise ValueError(f"unsupported quantization bit width: {quantized.bits}")
    return values.reshape(quantized.shape).to(dtype)


class QuantizedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, bits: int = 4) -> None:
        super().__init__()
        if bits not in (4, 8):
            raise ValueError("QuantizedLinear supports 4 or 8 bits")
        quantized = quantize_nf4(linear.weight) if bits == 4 else quantize_int8(linear.weight)
        self.register_buffer("codes", quantized.codes)
        self.register_buffer("scales", quantized.scales)
        self.shape = quantized.shape
        self.bits = bits
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = dequantize(
            QuantizedWeight(self.codes, self.scales, self.shape, self.bits, 64), inputs.dtype
        )
        return torch.nn.functional.linear(inputs, weight, self.bias)
