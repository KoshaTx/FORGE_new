"""Packed host batches and CUDA gradient results for family workers."""

import torch


def pack_batch(batch):
    groups, layout = {}, []
    for key, value in batch.items():
        if value.device.type != "cpu" or value.dtype not in (torch.int64, torch.bool):
            raise ValueError("IPC batch requires host int64 and bool tensors")
        groups.setdefault(value.dtype, []).append(value.reshape(-1))
        layout.append((key, value.dtype, tuple(value.shape), value.numel()))
    return {dtype: torch.cat(values) for dtype, values in groups.items()}, layout


def unpack_batch(packed):
    groups, layout = packed
    offsets = dict.fromkeys(groups, 0)
    result = {}
    for key, dtype, shape, size in layout:
        offset = offsets[dtype]
        result[key] = groups[dtype][offset : offset + size].view(shape)
        offsets[dtype] += size
    if any(offsets[k] != v.numel() for k, v in groups.items()):
        raise ValueError("IPC batch layout does not cover its buffers")
    return result


def pack_result(result):
    gradient, available, loss, metrics = result
    if not all(torch.is_tensor(v) and v.numel() == 1 for v in metrics.values()):
        raise ValueError("Packed gradient metrics must be scalar tensors")
    # Counts are bounded by batch/node support and exactly representable in float32.
    scalars = [loss.reshape(1)] + [v.reshape(1).to(gradient.dtype) for v in metrics.values()]
    return (
        torch.cat([gradient, *scalars]),
        available,
        gradient.numel(),
        [(key, value.dtype) for key, value in metrics.items()],
    )


def unpack_result(packed, device):
    vector, available, size, layout = packed
    if vector.numel() != size + 1 + len(layout):
        raise ValueError("IPC result layout does not cover its buffer")
    local = vector.to(device)
    return (
        local[:size],
        available,
        local[size],
        {name: local[size + 1 + index].to(dtype) for index, (name, dtype) in enumerate(layout)},
    )
