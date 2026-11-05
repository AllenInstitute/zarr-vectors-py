"""The CUDA kernels behind device-side cell decode.

Three small kernels, compiled on first use through cupy's NVRTC path:

- :func:`zstd_walk` checks every zstd frame's structure -- header, each
  block header, the end -- before nvCOMP is allowed to see it (nvCOMP
  does not validate its input, and a truncated frame can hang its
  kernel);
- :func:`unframe` reads each cell's vlen-bytes frame (item count, byte
  length) and, for a ragged link cell, its group header, and reports
  where the payload starts, how long it is, and whether the frame was
  well formed;
- :func:`gather` copies many byte ranges, each at its own address, into
  one contiguous buffer at given offsets.

Segments are addressed by raw device pointers rather than offsets into a
single buffer, because decompressed cells arrive as one allocation each.
Every multi-byte field is read byte by byte: cell payloads carry no
alignment guarantee.
"""

from __future__ import annotations

import functools
from typing import Any

import cupy as cp
import numpy as np

_SOURCE = r"""
typedef unsigned long long u64;
typedef long long i64;
typedef unsigned char u8;

__device__ unsigned int ld_u32(const u8* p) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8)
         | ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}

__device__ i64 ld_i64(const u8* p) {
    u64 v = 0;
    for (int b = 7; b >= 0; --b) v = (v << 8) | (u64)p[b];
    return (i64)v;
}

// status: 0 ok, 1 frame shorter than its header, 2 item count is not 1,
// 3 frame length disagrees with its header, 4 bad ragged group header.
extern "C" __global__
void zv_unframe(const u64* ptrs, const i64* sizes, int n, const unsigned char* ragged_of,
                u64* payload, i64* length, int* status) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u8* p = (const u8*)ptrs[i];
    i64 size = sizes[i];
    payload[i] = ptrs[i];
    length[i] = 0;
    status[i] = 0;
    if (size == 0) return;                       // nothing stored: empty
    if (size < 8) { status[i] = 1; return; }
    if (ld_u32(p) != 1u) { status[i] = 2; return; }
    i64 len = (i64)ld_u32(p + 4);
    if (8 + len != size) { status[i] = 3; return; }
    const u8* body = p + 8;
    if (ragged_of[i]) {
        // int64 group count k, then k int64 byte offsets, then the rows.
        if (len < 8) { payload[i] = (u64)body; return; }   // no groups
        i64 k = ld_i64(body);
        if (k < 0 || 8 * (1 + k) > len) { status[i] = 4; return; }
        if (k == 0) { payload[i] = (u64)body; return; }
        i64 head = 8 * (1 + k);
        i64 data = len - head;
        i64 prev = 0;
        for (i64 g = 0; g < k; ++g) {
            i64 off = ld_i64(body + 8 * (1 + g));
            if (off < prev || off > data) { status[i] = 4; return; }
            prev = off;
        }
        i64 first = ld_i64(body + 8);
        payload[i] = (u64)(body + head + first);
        length[i] = data - first;
        return;
    }
    payload[i] = (u64)body;
    length[i] = len;
}

// Walk one zstd frame's structure without decoding it: magic, frame
// header, then every block header to the last block, then the optional
// checksum, which must end exactly at the end of the stored bytes.
// status: 0 ok, 10 not a zstd frame, 11 reserved bit set, 12 header runs
// past the end, 13 uses a dictionary, 14 blocks run past the end or leave
// bytes over, 15 a block header is invalid, 16 block sizes contradict the
// declared content size, 17 no content size declared.
extern "C" __global__
void zv_zstd_walk(const u64* ptrs, const i64* sizes, int n, i64* declared, int* status) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u8* p = (const u8*)ptrs[i];
    i64 size = sizes[i];
    declared[i] = -1;
    status[i] = 0;
    if (size < 5 || ld_u32(p) != 0xFD2FB528u) { status[i] = 10; return; }
    unsigned int fhd = p[4];
    int fcs_flag = fhd >> 6, single = (fhd >> 5) & 1, checksum = (fhd >> 2) & 1;
    if ((fhd >> 3) & 1) { status[i] = 11; return; }
    i64 pos = 5;
    u64 window = 0;
    if (!single) {
        if (pos >= size) { status[i] = 12; return; }
        unsigned int wd = p[pos++];
        u64 base = 1ull << (10 + (wd >> 3));
        window = base + (base / 8) * (wd & 7);
    }
    int dsz = (fhd & 3) == 0 ? 0 : (fhd & 3) == 1 ? 1 : (fhd & 3) == 2 ? 2 : 4;
    if (pos + dsz > size) { status[i] = 12; return; }
    for (int b = 0; b < dsz; ++b) if (p[pos + b]) { status[i] = 13; return; }
    pos += dsz;
    int fsz = fcs_flag == 0 ? single : fcs_flag == 1 ? 2 : fcs_flag == 2 ? 4 : 8;
    if (pos + fsz > size) { status[i] = 12; return; }
    i64 fcs = -1;
    if (fsz) {
        u64 v = 0;
        for (int b = fsz - 1; b >= 0; --b) v = (v << 8) | (u64)p[pos + b];
        if (fsz == 2) v += 256;
        fcs = (i64)v;
    }
    pos += fsz;
    if (fcs < 0) { status[i] = 17; return; }
    if (single) window = (u64)fcs;
    u64 bmax = window < 131072ull ? window : 131072ull;
    i64 known = 0, compressed = 0;
    for (;;) {
        if (pos + 3 > size) { status[i] = 14; return; }
        unsigned int h = (unsigned int)p[pos] | ((unsigned int)p[pos + 1] << 8)
                       | ((unsigned int)p[pos + 2] << 16);
        pos += 3;
        int last = h & 1, type = (h >> 1) & 3;
        i64 bs = (i64)(h >> 3);
        if (type == 3 || (u64)bs > bmax) { status[i] = 15; return; }
        if (type == 1) { pos += 1; known += bs; }
        else { pos += bs; if (type == 0) known += bs; else compressed += 1; }
        if (pos > size) { status[i] = 14; return; }
        if (last) break;
    }
    if (checksum) pos += 4;
    if (pos != size) { status[i] = 14; return; }
    if (known > fcs || fcs > known + compressed * (i64)bmax) { status[i] = 16; return; }
    declared[i] = fcs;
}

extern "C" __global__
void zv_gather(const u64* ptrs, const i64* lengths, const i64* dest, int n, u8* out) {
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        const u8* src = (const u8*)ptrs[i];
        u8* dst = out + dest[i];
        i64 len = lengths[i];
        for (i64 b = threadIdx.x; b < len; b += blockDim.x) dst[b] = src[b];
    }
}
"""


@functools.cache
def _module() -> Any:
    return cp.RawModule(code=_SOURCE)


def _launch(name: str, n: int, args: tuple[Any, ...], *, per_cell_block: bool = False) -> None:
    if n == 0:
        return
    kernel = _module().get_function(name)
    if per_cell_block:
        kernel((min(n, 65_535),), (256,), args)
    else:
        threads = 128
        kernel(((n + threads - 1) // threads,), (threads,), args)


def unframe(
    ptrs: Any, sizes: Any, ragged: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Payload addresses, lengths and statuses of vlen frames, on the host.

    ``ragged[i]`` marks a ragged link cell, whose group header is skipped.
    """
    n = int(ptrs.size)
    payload = cp.empty(n, dtype=cp.uint64)
    length = cp.empty(n, dtype=cp.int64)
    status = cp.empty(n, dtype=cp.int32)
    _launch("zv_unframe", n, (
        ptrs, sizes, np.int32(n), cp.asarray(ragged, dtype=cp.uint8), payload, length, status,
    ))
    return payload.get(), length.get(), status.get()


def zstd_walk(ptrs: Any, sizes: Any) -> tuple[np.ndarray, np.ndarray]:
    """Declared content sizes and structural statuses of zstd frames (host)."""
    n = int(ptrs.size)
    declared = cp.empty(n, dtype=cp.int64)
    status = cp.empty(n, dtype=cp.int32)
    _launch("zv_zstd_walk", n, (ptrs, sizes, np.int32(n), declared, status))
    return declared.get(), status.get()


def gather(ptrs: Any, lengths: np.ndarray, dest: np.ndarray, total: int) -> Any:
    """Copy segment ``i`` (``lengths[i]`` bytes) to ``out[dest[i]:]``."""
    out = cp.empty(int(total), dtype=cp.uint8)
    n = int(ptrs.size)
    _launch("zv_gather", n, (
        ptrs, cp.asarray(lengths, dtype=cp.int64), cp.asarray(dest, dtype=cp.int64),
        np.int32(n), out,
    ), per_cell_block=True)
    return out


#: Human-readable reasons for :func:`unframe` statuses.
UNFRAME_ERRORS = {
    1: "stored frame is shorter than its vlen-bytes header",
    2: "vlen-bytes frame does not hold exactly one item",
    3: "vlen-bytes frame length disagrees with its header",
    4: "ragged link cell has a malformed group header",
    10: "stored bytes are not a zstd frame",
    11: "zstd frame header sets a reserved bit",
    12: "zstd frame header runs past the stored bytes",
    13: "zstd frame needs a dictionary",
    14: "zstd frame is truncated or has trailing bytes",
    15: "zstd frame has an invalid block header",
    16: "zstd frame's blocks contradict its declared content size",
    17: "zstd frame does not declare its content size",
}
