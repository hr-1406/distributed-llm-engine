from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


def _world(group: dist.ProcessGroup | None) -> tuple[int, int]:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 1, 0
    return dist.get_world_size(group), dist.get_rank(group)


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, group: dist.ProcessGroup | None = None, bias: bool = False) -> None:
        super().__init__()
        world, _ = _world(group)
        if out_features % world:
            raise ValueError("out_features must be divisible by tensor parallel world size")
        self.group = group
        self.out_features_per_rank = out_features // world
        self.weight = nn.Parameter(torch.empty(self.out_features_per_rank, in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features_per_rank)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(hidden_states, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, group: dist.ProcessGroup | None = None, bias: bool = True) -> None:
        super().__init__()
        world, _ = _world(group)
        if in_features % world:
            raise ValueError("in_features must be divisible by tensor parallel world size")
        self.group = group
        self.in_features_per_rank = in_features // world
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_rank))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        local = torch.nn.functional.linear(hidden_states, self.weight, None)
        if self.group is not None and dist.is_available() and dist.is_initialized():
            dist.all_reduce(local, group=self.group)
        return local + self.bias if self.bias is not None else local


class TensorParallelMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, group: dist.ProcessGroup | None = None) -> None:
        super().__init__()
        self.gate = ColumnParallelLinear(hidden_size, intermediate_size, group, bias=False)
        self.up = ColumnParallelLinear(hidden_size, intermediate_size, group, bias=False)
        self.down = RowParallelLinear(intermediate_size, hidden_size, group, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = torch.nn.functional.silu(self.gate(hidden_states)) * self.up(hidden_states)
        return self.down(gated)
