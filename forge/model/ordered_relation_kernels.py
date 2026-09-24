"""FP32 embedding reduction with PyTorch 2.13 CUDA's ordered groups of ten."""

import triton
import triton.language as tl


@triton.jit
def partial_sums(grad, order, bounds, offsets, partials, heads: tl.constexpr, block: tl.constexpr):
    relation = tl.program_id(1)
    begin = tl.load(bounds + relation)
    end = tl.load(bounds + relation + 1)
    count = (end - begin + 9) // 10
    chunk = tl.program_id(0) * block + tl.arange(0, block)
    feature = tl.arange(0, heads)
    if tl.program_id(0) * block < count:
        acc = tl.full((block, heads), 0, tl.float32)
        for row in tl.static_range(10):
            position = begin + chunk * 10 + row
            index = tl.load(order + position, position < end, other=0)
            value = tl.load(
                grad + index[:, None] * heads + feature[None, :],
                position[:, None] < end,
                other=0,
            )
            acc = acc + value
        offset = tl.load(offsets + relation)
        tl.store(
            partials + (offset + chunk[:, None]) * heads + feature[None, :],
            acc,
            chunk[:, None] < count,
        )


@triton.jit
def ordered_sum(partials, offsets, output, heads: tl.constexpr, tile: tl.constexpr):
    relation = tl.program_id(0)
    begin = tl.load(offsets + relation)
    end = tl.load(offsets + relation + 1)
    feature = tl.arange(0, heads)
    lane = tl.arange(0, tile)
    acc = tl.full((heads, 1), 0, tl.float32)
    # Load independent rows together, then add in the original sequence. A parallel
    # tree reduction here would reintroduce the rounding drift this kernel addresses.
    for start in range(begin, end, tile):
        values = tl.load(
            partials + (start + lane[None, :]) * heads + feature[:, None],
            start + lane[None, :] < end,
            other=0,
        )
        for column in tl.static_range(tile):
            value = tl.gather(values, tl.full((heads, 1), column, tl.int32), 1)
            acc = acc + value
    tl.store(output + relation * heads + feature[:, None], acc)
