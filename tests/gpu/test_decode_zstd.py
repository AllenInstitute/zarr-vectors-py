"""``zarr_vectors.gpu.decode_zstd``: checked zstd decode for callers' own bytes.

The frames are what zarr writes (numcodecs' zstd), decoded on the device
and compared with the bytes that went in. Damaged frames must come back
as errors -- quickly, not as a hung kernel or a dead CUDA context.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.gpu._cuda import CUDA, cupy

pytest.importorskip("nvidia.nvcomp")

from numcodecs import Zstd  # noqa: E402

from zarr_vectors.gpu import decode_zstd  # noqa: E402

pytestmark = CUDA


def _raw(n):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 7, 4000 + 97 * i, dtype=np.uint8).tobytes() for i in range(n)]


def _frames(raw):
    return [cupy.asarray(np.frombuffer(Zstd(level=3).encode(r), np.uint8)) for r in raw]


def test_frames_decode_to_what_went_in():
    raw = _raw(5)
    outputs, errors = decode_zstd(_frames(raw))
    assert errors == {}
    for o, r in zip(outputs, raw):
        assert o.dtype == cupy.uint8 and o.get().tobytes() == r


def test_nothing_in_nothing_out():
    assert decode_zstd([]) == ([], {})


def test_a_frame_declaring_another_size_is_reported():
    raw = _raw(3)
    want = [len(r) for r in raw]
    want[1] += 1
    outputs, errors = decode_zstd(_frames(raw), want)
    assert set(errors) == {1} and "expected" in errors[1] and outputs[1] is None
    assert outputs[0].get().tobytes() == raw[0]
    assert outputs[2].get().tobytes() == raw[2]


def test_one_expected_size_applies_to_every_frame():
    raw = [bytes(range(256)) * 16] * 3
    outputs, errors = decode_zstd(_frames(raw), 4096)
    assert errors == {} and all(o.get().tobytes() == raw[0] for o in outputs)


def test_decoding_into_the_callers_buffers():
    raw = _raw(4)
    out = [cupy.empty(len(r), dtype=cupy.uint8) for r in raw]
    outputs, errors = decode_zstd(_frames(raw), out=out)
    assert errors == {}
    for o, b, r in zip(outputs, out, raw):
        assert o.data.ptr == b.data.ptr
        assert b.get().tobytes() == r


def test_an_output_of_the_wrong_size_is_reported():
    raw = _raw(2)
    out = [cupy.empty(len(raw[0]), dtype=cupy.uint8), cupy.empty(len(raw[1]) - 1, dtype=cupy.uint8)]
    outputs, errors = decode_zstd(_frames(raw), out=out)
    assert set(errors) == {1} and "its output holds" in errors[1]
    assert out[0].get().tobytes() == raw[0]


@pytest.mark.parametrize("damage", ["truncated", "trailing", "not zstd", "empty"])
def test_a_damaged_frame_is_an_error_and_the_rest_decode(damage):
    raw = _raw(3)
    frames = _frames(raw)
    bad = frames[1].get().tobytes()
    bad = {
        "truncated": bad[: len(bad) // 2],
        "trailing": bad + b"\0\0\0",
        "not zstd": bytes(range(200)),
        "empty": b"",
    }[damage]
    frames[1] = cupy.asarray(np.frombuffer(bad, np.uint8))
    outputs, errors = decode_zstd(frames)
    assert set(errors) == {1} and outputs[1] is None
    assert outputs[0].get().tobytes() == raw[0]
    assert outputs[2].get().tobytes() == raw[2]


def test_it_runs_on_the_callers_stream():
    raw = _raw(3)
    frames = _frames(raw)
    # The frames were copied up on the default stream, which a non-blocking
    # stream does not wait for: order them first, as a caller must.
    cupy.cuda.get_current_stream().synchronize()
    stream = cupy.cuda.Stream(non_blocking=True)
    out = [cupy.empty(len(r), dtype=cupy.uint8) for r in raw]
    outputs, errors = decode_zstd(frames, out=out, stream=stream)
    # Returned synchronised: readable from another stream straight away.
    assert errors == {} and [o.get().tobytes() for o in outputs] == raw


def test_a_strided_frame_is_refused():
    frame = _frames(_raw(1))[0]
    with pytest.raises(ValueError, match="not C-contiguous"):
        decode_zstd([frame[::2]])


def test_a_sharded_zarr_arrays_chunks_decode_from_their_stored_ranges(tmp_path):
    """The layout zvDVC's converter writes: bytes + zstd inside shards."""
    import zarr
    from zarr.codecs import ZstdCodec

    data = np.random.default_rng(3).integers(0, 4000, (64, 64, 64), dtype=np.uint16)
    arr = zarr.create_array(
        str(tmp_path / "v.zarr"), shape=data.shape, dtype=data.dtype,
        chunks=(16, 16, 16), shards=(32, 32, 32), compressors=ZstdCodec(level=3),
    )
    arr[...] = data
    shard = tmp_path / "v.zarr" / "c" / "0" / "0" / "0"
    blob = shard.read_bytes()
    n_inner = 8
    index = np.frombuffer(blob[-(n_inner * 16 + 4):-4], dtype="<u8").reshape(n_inner, 2)
    frames = [
        cupy.asarray(np.frombuffer(blob[int(o):int(o) + int(s)], np.uint8)) for o, s in index
    ]
    chunk_nbytes = 16 * 16 * 16 * 2
    outputs, errors = decode_zstd(frames, chunk_nbytes)
    assert errors == {}
    for k, o in enumerate(outputs):
        z, y, x = np.unravel_index(k, (2, 2, 2))
        want = data[z * 16:(z + 1) * 16, y * 16:(y + 1) * 16, x * 16:(x + 1) * 16]
        got = o.view(cupy.uint16).reshape(16, 16, 16).get()
        np.testing.assert_array_equal(got, want)
