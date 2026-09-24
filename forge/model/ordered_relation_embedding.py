"""Optional H100 FP32 relation lookup with ordered deterministic gradient reduction."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

BACKEND = "ordered_fp32_v1"


def ordered_backward(grad, indices, rows):
    if grad.device.type != "cuda" or grad.dtype != torch.float32:
        raise ValueError("ordered embedding requires CUDA FP32")
    if rows != 5 or grad.shape[-1] != 8 or indices.dtype != torch.int64:
        raise ValueError("ordered embedding requires five relations, eight heads and int64 indices")
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError("ordered embedding is qualified against deterministic reduction only")
    if indices.numel() <= 3072:
        return torch.ops.aten.embedding_dense_backward(grad, indices, rows, -1, False)
    from forge.model.ordered_relation_kernels import ordered_sum, partial_sums

    flat = indices.reshape(-1)
    sorted_indices, order = torch.sort(flat.to(torch.uint8), stable=True)
    bounds = torch.searchsorted(
        sorted_indices, torch.arange(rows + 1, device=flat.device, dtype=torch.uint8)
    )
    counts = (bounds[1:] - bounds[:-1] + 9) // 10
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    partials = grad.new_empty((flat.numel() // 10 + rows, 8))
    output = grad.new_empty((rows, 8))
    partial_sums[((flat.numel() + 1279) // 1280, rows)](
        grad.contiguous(),
        order,
        bounds,
        offsets,
        partials,
        8,
        128,
        enable_fp_fusion=False,
    )
    ordered_sum[(rows,)](
        partials,
        offsets,
        output,
        8,
        128,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return output


class _OrderedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, indices):
        ctx.save_for_backward(indices)
        ctx.rows = weight.shape[0]
        return functional.embedding(indices, weight)

    @staticmethod
    def backward(ctx, grad):
        (indices,) = ctx.saved_tensors
        return ordered_backward(grad, indices, ctx.rows), None


class OrderedRelationEmbedding(torch.nn.Module):
    """Keep the original Parameter object and state-dict key when enabling the kernel."""

    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, indices):
        return _OrderedEmbedding.apply(self.weight, indices)


def configure_relation_embedding(model, backend=None):
    """Enable only the explicitly qualified backend; default execution is unchanged."""
    if backend is None:
        return
    if backend != BACKEND:
        raise ValueError(f"Unknown relation_embedding_backend: {backend}")
    if isinstance(model.relation_bias, OrderedRelationEmbedding):
        return
    weight = model.relation_bias.weight
    if weight.device.type != "cuda" or weight.dtype != torch.float32 or weight.shape != (5, 8):
        raise ValueError(
            "Ordered relation embedding requires CUDA FP32 with five states/eight heads"
        )
    if torch.__version__ != "2.13.0+cu130" or "H100" not in torch.cuda.get_device_name(
        weight.device
    ):
        raise ValueError("Ordered relation embedding is qualified for PyTorch 2.13.0+cu130 on H100")
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError("Ordered relation embedding requires deterministic algorithms")
    model.relation_bias = OrderedRelationEmbedding(weight)
