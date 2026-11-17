"""The CUDA kernels behind device-side cell decode, compiled on first use.

- :func:`zstd_walk` checks each zstd frame's structure before nvCOMP sees it;
- :func:`unframe` parses each cell's vlen-bytes frame (and a ragged link
  cell's group header), reporting payload address, length and status;
- :func:`gather` copies many byte ranges into one contiguous buffer.

Segments are addressed by raw device pointers (decompressed cells are one
allocation each), and multi-byte fields are read byte by byte (payloads
carry no alignment guarantee).
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
void zv_zstd_walk(const u64* ptrs, const i64* sizes, int n, i64* declared, int* status,
                  u8* has_checksum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u8* p = (const u8*)ptrs[i];
    i64 size = sizes[i];
    declared[i] = -1;
    status[i] = 0;
    has_checksum[i] = 0;
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
    has_checksum[i] = (u8)checksum;
}

// XXH64 (seed 0), the hash a zstd frame's content checksum is the low 32
// bits of (RFC 8878, 3.1.1). One thread per buffer: the hash is serial.
#define P1 0x9E3779B185EBCA87ull
#define P2 0xC2B2AE3D27D4EB4Full
#define P3 0x165667B19E3779F9ull
#define P4 0x85EBCA77C2B2AE63ull
#define P5 0x27D4EB2F165667C5ull

__device__ u64 rotl64(u64 x, int r) { return (x << r) | (x >> (64 - r)); }
__device__ u64 xxh_round(u64 acc, u64 v) { acc += v * P2; return rotl64(acc, 31) * P1; }
__device__ u64 xxh_merge(u64 acc, u64 v) { acc ^= xxh_round(0, v); return acc * P1 + P4; }
__device__ u64 ld_u64(const u8* p, bool aligned) {
    return aligned ? *(const u64*)p : (u64)ld_i64(p);
}

__device__ u64 xxh64(const u8* p, i64 len) {
    const u8* end = p + len;
    bool aligned = ((u64)p & 7) == 0;
    u64 h;
    if (len >= 32) {
        u64 v1 = P1 + P2, v2 = P2, v3 = 0, v4 = 0 - P1;
        const u8* limit = end - 32;
        do {
            v1 = xxh_round(v1, ld_u64(p, aligned));
            v2 = xxh_round(v2, ld_u64(p + 8, aligned));
            v3 = xxh_round(v3, ld_u64(p + 16, aligned));
            v4 = xxh_round(v4, ld_u64(p + 24, aligned));
            p += 32;
        } while (p <= limit);
        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = xxh_merge(h, v1); h = xxh_merge(h, v2);
        h = xxh_merge(h, v3); h = xxh_merge(h, v4);
    } else {
        h = P5;
    }
    h += (u64)len;
    while (p + 8 <= end) {
        h ^= xxh_round(0, ld_u64(p, aligned));
        h = rotl64(h, 27) * P1 + P4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= (u64)ld_u32(p) * P1;
        h = rotl64(h, 23) * P2 + P3;
        p += 4;
    }
    while (p < end) {
        h ^= (u64)(*p) * P5;
        h = rotl64(h, 11) * P1;
        ++p;
    }
    h ^= h >> 33; h *= P2;
    h ^= h >> 29; h *= P3;
    h ^= h >> 32;
    return h;
}

extern "C" __global__
void zv_xxh64(const u64* ptrs, const i64* sizes, int n, u64* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = xxh64((const u8*)ptrs[i], sizes[i]);
}

// Decoded buffer i against the 4-byte content checksum ending its frame:
// ok[i] = 1 when the low 32 bits of its XXH64 match.
extern "C" __global__
void zv_zstd_checksum(const u64* ptrs, const i64* sizes, const u64* frame_ends, int n,
                      u8* ok) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int stored = ld_u32((const u8*)frame_ends[i] - 4);
    ok[i] = (unsigned int)(xxh64((const u8*)ptrs[i], sizes[i]) & 0xFFFFFFFFull) == stored;
}

// Check one ZVFG fragment-index blob (zarr_vectors.encoding.fragments)
// and report its header and section sizes. status: 0 ok (or empty),
// 20 shorter than the header, 21 bad magic, 22 bad version, 23 flags set,
// 24 R > F, 25-28 truncated in the bitmap / range table / explicit offsets /
// explicit indices, 29 the bitmap's set bits are not R, 30 the explicit
// offsets do not start at 0 or decrease.
extern "C" __global__
void zv_frag_header(const u64* ptrs, const i64* sizes, int n, i64* head, int* status) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u8* p = (const u8*)ptrs[i];
    i64 size = sizes[i];
    i64* h = head + 6 * i;          // magic, version, flags, F, R, T
    for (int k = 0; k < 6; ++k) h[k] = 0;
    status[i] = 0;
    if (size == 0) return;
    if (size < 16) { status[i] = 20; return; }
    unsigned int magic = ld_u32(p);
    unsigned int version = (unsigned int)p[4] | ((unsigned int)p[5] << 8);
    unsigned int flags = (unsigned int)p[6] | ((unsigned int)p[7] << 8);
    i64 f = (i64)ld_u32(p + 8), r = (i64)ld_u32(p + 12);
    h[0] = magic; h[1] = version; h[2] = flags; h[3] = f; h[4] = r;
    if (magic != 0x5A564647u) { status[i] = 21; return; }
    if (version != 1u) { status[i] = 22; return; }
    if (flags != 0u) { status[i] = 23; return; }
    if (r > f) { status[i] = 24; return; }
    if (f == 0) { h[3] = 0; h[4] = 0; return; }
    i64 pos = 16;
    i64 raw_bitmap = (f + 7) / 8;
    i64 bitmap = (raw_bitmap + 7) & ~(i64)7;
    if (size < pos + bitmap) { status[i] = 25; return; }
    i64 bits = 0;
    for (i64 b = 0; b < raw_bitmap; ++b) {
        unsigned int byte = p[pos + b];
        if (b == raw_bitmap - 1 && (f & 7)) byte &= (1u << (f & 7)) - 1u;
        bits += __popc(byte);
    }
    pos += bitmap;
    if (size < pos + 16 * r) { status[i] = 26; return; }
    pos += 16 * r;
    i64 e = f - r;
    if (size < pos + 4 * (e + 1)) { status[i] = 27; return; }
    unsigned int prev = ld_u32(p + pos);
    bool monotone = prev == 0u;
    for (i64 k = 1; k <= e && monotone; ++k) {
        unsigned int cur = ld_u32(p + pos + 4 * k);
        monotone = cur >= prev;
        prev = cur;
    }
    i64 t = e > 0 ? (i64)ld_u32(p + pos + 4 * e) : 0;
    pos += 4 * (e + 1);
    if (size < pos + 8 * t) { status[i] = 28; return; }
    if (bits != r) { status[i] = 29; return; }
    if (!monotone) { status[i] = 30; return; }
    h[5] = t;
}

// One thread per output fragment g of cell c = cell[g], local index
// f = g - first[c]: is fragment f of its cell a range?
extern "C" __global__
void zv_frag_bits(const int* cell, const i64* first, const u64* bitmap, i64 n, u8* is_range) {
    i64 g = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    int c = cell[g];
    i64 f = g - first[c];
    const u8* bm = (const u8*)bitmap[c];
    is_range[g] = (bm[f >> 3] >> (f & 7)) & 1;
}

// One thread per output fragment: its (start, count, index count), from
// the range table row ``rank[g]`` (range fragments before it in its cell)
// or the explicit offsets of explicit fragment f - rank[g].
extern "C" __global__
void zv_frag_fill(const int* cell, const i64* first, const u64* table, const u64* offsets,
                  const u8* is_range, const i64* rank, i64 n,
                  i64* starts, i64* counts, i64* index_counts) {
    i64 g = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    int c = cell[g];
    i64 f = g - first[c];
    i64 r = rank[g];
    if (is_range[g]) {
        const u8* row = (const u8*)table[c] + 16 * r;
        starts[g] = ld_i64(row);
        counts[g] = ld_i64(row + 8);
        index_counts[g] = 0;
    } else {
        const u8* off = (const u8*)offsets[c] + 4 * (f - r);
        i64 count = (i64)ld_u32(off + 4) - (i64)ld_u32(off);
        starts[g] = -1;
        counts[g] = count;
        index_counts[g] = count;
    }
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


def zstd_walk(ptrs: Any, sizes: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Declared content sizes, structural statuses and content-checksum
    flags of zstd frames (host)."""
    n = int(ptrs.size)
    declared = cp.empty(n, dtype=cp.int64)
    status = cp.empty(n, dtype=cp.int32)
    has_checksum = cp.zeros(n, dtype=cp.uint8)
    _launch("zv_zstd_walk", n, (ptrs, sizes, np.int32(n), declared, status, has_checksum))
    return declared.get(), status.get(), has_checksum.get().astype(bool)


def xxh64(ptrs: Any, sizes: Any) -> np.ndarray:
    """XXH64 (seed 0) of each device buffer, on the host (``uint64``)."""
    n = int(ptrs.size)
    out = cp.empty(n, dtype=cp.uint64)
    _launch("zv_xxh64", n, (ptrs, sizes, np.int32(n), out))
    return out.get()


def zstd_checksums_match(ptrs: Any, sizes: Any, frame_ends: Any) -> np.ndarray:
    """Whether each decoded buffer matches the content checksum in the last
    four bytes before ``frame_ends[i]`` (host, ``bool``)."""
    n = int(ptrs.size)
    ok = cp.zeros(n, dtype=cp.uint8)
    _launch("zv_zstd_checksum", n, (ptrs, sizes, frame_ends, np.int32(n), ok))
    return ok.get().astype(bool)


def gather(ptrs: Any, lengths: np.ndarray, dest: np.ndarray, total: int) -> Any:
    """Copy segment ``i`` (``lengths[i]`` bytes) to ``out[dest[i]:]``."""
    out = cp.empty(int(total), dtype=cp.uint8)
    n = int(ptrs.size)
    _launch("zv_gather", n, (
        ptrs, cp.asarray(lengths, dtype=cp.int64), cp.asarray(dest, dtype=cp.int64),
        np.int32(n), out,
    ), per_cell_block=True)
    return out


def fragment_headers(ptrs: Any, sizes: Any) -> tuple[np.ndarray, np.ndarray]:
    """Each ZVFG blob's ``(magic, version, flags, F, R, T)`` row and status (host)."""
    n = int(ptrs.size)
    head = cp.zeros((n, 6), dtype=cp.int64)
    status = cp.zeros(n, dtype=cp.int32)
    _launch("zv_frag_header", n, (ptrs, sizes, np.int32(n), head, status))
    return head.get(), status.get()


def fragment_rows(
    first: Any, bitmap: Any, table: Any, offsets: Any, total: int,
) -> tuple[Any, Any, Any]:
    """``(starts, counts, index_counts)`` of every fragment of every cell.

    Cell ``c``'s fragments are output rows ``first[c]:first[c + 1]``;
    ``bitmap``, ``table`` and ``offsets`` are the device addresses of its
    range bitmap, range table and explicit offsets.
    """
    n = int(total)
    starts = cp.empty(n, dtype=cp.int64)
    counts = cp.empty(n, dtype=cp.int64)
    index_counts = cp.empty(n, dtype=cp.int64)
    if n == 0:
        return starts, counts, index_counts
    g = cp.arange(n, dtype=cp.int64)
    cell = (cp.searchsorted(first, g, side="right") - 1).astype(cp.int32)
    is_range = cp.empty(n, dtype=cp.uint8)
    blocks = ((n + 127) // 128,)
    module = _module()
    module.get_function("zv_frag_bits")(
        blocks, (128,), (cell, first, bitmap, np.int64(n), is_range),
    )
    # Range fragments before each one, within its cell.
    before = cp.cumsum(is_range, dtype=cp.int64) - is_range
    # (An empty cell's first row may be n; no fragment looks it up.)
    rank = before - before[cp.minimum(first[:-1], n - 1)][cell]
    module.get_function("zv_frag_fill")(blocks, (128,), (
        cell, first, table, offsets, is_range, rank, np.int64(n),
        starts, counts, index_counts,
    ))
    return starts, counts, index_counts


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
    18: "zstd frame's content checksum does not match its decoded bytes",
}
