from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BlockHandle:
    block_id: int
    generation: int


class KVBlockPool:
    """Thread-safe fixed-size KV allocator with generation-checked handles."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
    ) -> None:
        if min(num_blocks, block_size, num_layers, num_kv_heads, head_dim) <= 0:
            raise ValueError("all KV pool dimensions must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free = list(range(num_blocks - 1, -1, -1))
        self._generation = [0] * num_blocks
        self._owners: dict[int, str] = {}
        self._lock = threading.RLock()
        resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        shape = (2, num_blocks, num_layers, block_size, num_kv_heads, head_dim)
        self.cache = torch.empty(shape, dtype=dtype, device=resolved_device)

    def allocate(self, owner: str, count: int = 1) -> list[BlockHandle]:
        if not owner or count <= 0:
            raise ValueError("owner must be non-empty and count must be positive")
        with self._lock:
            if count > len(self._free):
                raise MemoryError(f"KV cache exhausted: requested={count}, free={len(self._free)}")
            handles = []
            for _ in range(count):
                block_id = self._free.pop()
                self._generation[block_id] += 1
                self._owners[block_id] = owner
                handles.append(BlockHandle(block_id, self._generation[block_id]))
            return handles

    def release(self, handles: Iterable[BlockHandle], owner: str) -> None:
        with self._lock:
            for handle in handles:
                if self._owners.get(handle.block_id) != owner:
                    raise ValueError("cannot release a block owned by another request")
                if self._generation[handle.block_id] != handle.generation:
                    raise ValueError("stale KV block handle")
            for handle in handles:
                del self._owners[handle.block_id]
                self._free.append(handle.block_id)

    def write(
        self,
        handle: BlockHandle,
        layer: int,
        token_offset: int,
        key: torch.Tensor,
        value: torch.Tensor,
        owner: str,
    ) -> None:
        if not 0 <= token_offset < self.block_size:
            raise IndexError("token_offset outside KV block")
        with self._lock:
            if self._owners.get(handle.block_id) != owner:
                raise ValueError("block is not owned by this request")
            if self._generation[handle.block_id] != handle.generation:
                raise ValueError("stale KV block handle")
            self.cache[0, handle.block_id, layer, token_offset].copy_(key)
            self.cache[1, handle.block_id, layer, token_offset].copy_(value)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"total": self.num_blocks, "free": len(self._free), "used": len(self._owners)}


def paged_attention(
    query: torch.Tensor,
    block_table: torch.Tensor,
    context_lengths: torch.Tensor,
    kv_cache: torch.Tensor,
    block_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference paged attention; production deployments can swap in the Triton kernel."""
    if query.ndim != 4 or block_table.ndim != 2 or kv_cache.ndim != 6:
        raise ValueError("query, block_table, and kv_cache have invalid ranks")
    batch, query_heads, _, head_dim = query.shape
    _, max_blocks = block_table.shape
    if context_lengths.shape != (batch,):
        raise ValueError("context_lengths must have shape [batch]")
    keys, values = kv_cache[0], kv_cache[1]
    outputs = []
    scale_value = scale or head_dim**-0.5
    for batch_id in range(batch):
        length = int(context_lengths[batch_id].item())
        if length <= 0 or length > max_blocks * block_size:
            raise ValueError("context length exceeds block table capacity")
        token_keys = []
        token_values = []
        for token in range(length):
            block = int(block_table[batch_id, token // block_size].item())
            offset = token % block_size
            token_keys.append(keys[block, 0, offset])
            token_values.append(values[block, 0, offset])
        key_matrix = torch.stack(token_keys).transpose(0, 1)
        value_matrix = torch.stack(token_values).transpose(0, 1)
        scores = torch.einsum("hd,htd->ht", query[batch_id, :, 0], key_matrix) * scale_value
        outputs.append(torch.einsum("ht,htd->hd", torch.softmax(scores, -1), value_matrix).unsqueeze(1))
    return torch.cat(outputs, dim=0).reshape(batch, query_heads, 1, head_dim)
