import pytest
import torch
from engine.memory.paged_attention import KVBlockPool, paged_attention
from engine.quantization.bitsandbytes_quant import dequantize, quantize_int8, quantize_nf4


def test_kv_pool_lifecycle():
    pool = KVBlockPool(2, 4, 1, 1, 8, device="cpu")
    handle = pool.allocate("request-a")[0]
    assert pool.stats() == {"total": 2, "free": 1, "used": 1}
    key = torch.ones(1, 8)
    pool.write(handle, 0, 0, key, key, "request-a")
    pool.release([handle], "request-a")
    assert pool.stats()["free"] == 2
    with pytest.raises(ValueError):
        pool.release([handle], "request-a")


def test_reference_paged_attention_shape():
    cache = torch.randn(2, 2, 1, 4, 1, 8)
    query = torch.randn(1, 1, 1, 8)
    result = paged_attention(query, torch.tensor([[0]]), torch.tensor([2]), cache, 4)
    assert result.shape == query.shape


def test_quantization_round_trips():
    weight = torch.randn(32, 16)
    int8 = dequantize(quantize_int8(weight), torch.float32)
    nf4 = dequantize(quantize_nf4(weight, block_size=64), torch.float32)
    assert int8.shape == weight.shape
    assert nf4.shape == weight.shape
    assert torch.mean((weight - int8) ** 2) < 0.01
