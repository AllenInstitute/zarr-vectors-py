"""Draco mesh/point cloud compression wrapper.

Tries DracoPy first; falls back to subprocess calls to
``draco_encoder`` / ``draco_decoder`` CLI tools.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt

from zarr_vectors.exceptions import DracoError

_HAS_DRACOPY: bool | None = None


def _check_dracopy() -> bool:
    global _HAS_DRACOPY
    if _HAS_DRACOPY is None:
        try:
            import DracoPy  # noqa: F401
            _HAS_DRACOPY = True
        except ImportError:
            _HAS_DRACOPY = False
    return _HAS_DRACOPY


def draco_backend_available() -> bool:
    """True when DracoPy imports or a ``draco_encoder`` CLI is on PATH.

    Lets a writer refuse a Draco write before it creates anything,
    rather than fail on the first chunk with a half-written store.
    """
    if _check_dracopy():
        return True
    import shutil
    return shutil.which("draco_encoder") is not None


def draco_encode_mesh(
    positions: npt.NDArray[np.floating],
    faces: npt.NDArray[np.integer],
    *,
    quantization_bits: int = 11,
    compression_level: int = 7,
) -> bytes:
    """Encode a mesh to a Draco bitstream.

    Args:
        positions: ``(V, 3)`` float vertex positions.
        faces: ``(F, 3)`` int triangle face indices.
        quantization_bits: Position quantization (default 11).
        compression_level: 0–10 (default 7).

    Returns:
        Draco-compressed bytes.

    Raises:
        DracoError: If encoding fails.
    """
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.uint32)

    if _check_dracopy():
        return _encode_mesh_dracopy(positions, faces, quantization_bits, compression_level)

    return _encode_mesh_cli(positions, faces, quantization_bits, compression_level)


def draco_decode_mesh(data: bytes) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """Decode a Draco bitstream to mesh positions and faces.

    Returns:
        (positions, faces) — ``(V, 3)`` float32 and ``(F, 3)`` int64.

    Raises:
        DracoError: If decoding fails.
    """
    if _check_dracopy():
        return _decode_mesh_dracopy(data)
    return _decode_mesh_cli(data)


def draco_encode_point_cloud(
    positions: npt.NDArray[np.floating],
    *,
    quantization_bits: int = 11,
) -> bytes:
    """Encode point positions to a Draco bitstream.

    Args:
        positions: ``(N, 3)`` float vertex positions.
        quantization_bits: Position quantization (default 11).

    Returns:
        Draco-compressed bytes.
    """
    positions = np.ascontiguousarray(positions, dtype=np.float32)

    if _check_dracopy():
        return _encode_pc_dracopy(positions, quantization_bits)

    return _encode_pc_cli(positions, quantization_bits)


def draco_decode_point_cloud(data: bytes) -> npt.NDArray[np.float32]:
    """Decode a Draco bitstream to point cloud positions.

    Returns:
        ``(N, 3)`` float32 positions.
    """
    if _check_dracopy():
        return _decode_pc_dracopy(data)
    return _decode_pc_cli(data)


# ===================================================================
# DracoPy implementations
# ===================================================================

def _encode_mesh_dracopy(
    positions: npt.NDArray, faces: npt.NDArray,
    qbits: int, clevel: int,
) -> bytes:
    try:
        import DracoPy
        return DracoPy.encode(
            positions.flatten().tolist(),
            faces.flatten().tolist(),
            quantization_bits=qbits,
            compression_level=clevel,
        )
    except Exception as e:
        raise DracoError(f"DracoPy mesh encode failed: {e}") from e


def _decode_mesh_dracopy(data: bytes) -> tuple[npt.NDArray, npt.NDArray]:
    try:
        import DracoPy
        mesh = DracoPy.decode(data)
        positions = np.array(mesh.points, dtype=np.float32).reshape(-1, 3)
        faces = np.array(mesh.faces, dtype=np.int64).reshape(-1, 3)
        return positions, faces
    except Exception as e:
        raise DracoError(f"DracoPy mesh decode failed: {e}") from e


def _encode_pc_dracopy(positions: npt.NDArray, qbits: int) -> bytes:
    try:
        import DracoPy
        return DracoPy.encode(
            positions.flatten().tolist(),
            quantization_bits=qbits,
        )
    except Exception as e:
        raise DracoError(f"DracoPy point cloud encode failed: {e}") from e


def _decode_pc_dracopy(data: bytes) -> npt.NDArray:
    try:
        import DracoPy
        pc = DracoPy.decode(data)
        return np.array(pc.points, dtype=np.float32).reshape(-1, 3)
    except Exception as e:
        raise DracoError(f"DracoPy point cloud decode failed: {e}") from e


# ===================================================================
# CLI fallback implementations
# ===================================================================

def _encode_mesh_cli(
    positions: npt.NDArray, faces: npt.NDArray,
    qbits: int, clevel: int,
) -> bytes:
    """Encode via draco_encoder CLI by writing a temporary PLY."""
    try:
        with tempfile.TemporaryDirectory() as td:
            ply_path = Path(td) / "input.ply"
            drc_path = Path(td) / "output.drc"
            _write_temp_ply(ply_path, positions, faces)
            subprocess.run(
                ["draco_encoder", "-i", str(ply_path), "-o", str(drc_path),
                 "-qp", str(qbits), "-cl", str(clevel)],
                check=True, capture_output=True,
            )
            return drc_path.read_bytes()
    except FileNotFoundError:
        raise DracoError(
            "Neither DracoPy nor draco_encoder CLI found. "
            "Install with: pip install DracoPy"
        )
    except subprocess.CalledProcessError as e:
        raise DracoError(f"draco_encoder failed: {e.stderr.decode()}") from e


def _decode_mesh_cli(data: bytes) -> tuple[npt.NDArray, npt.NDArray]:
    try:
        with tempfile.TemporaryDirectory() as td:
            drc_path = Path(td) / "input.drc"
            ply_path = Path(td) / "output.ply"
            drc_path.write_bytes(data)
            subprocess.run(
                ["draco_decoder", "-i", str(drc_path), "-o", str(ply_path)],
                check=True, capture_output=True,
            )
            return _read_temp_ply(ply_path)
    except FileNotFoundError:
        raise DracoError(
            "Neither DracoPy nor draco_decoder CLI found. "
            "Install with: pip install DracoPy"
        )
    except subprocess.CalledProcessError as e:
        raise DracoError(f"draco_decoder failed: {e.stderr.decode()}") from e


def _encode_pc_cli(positions: npt.NDArray, qbits: int) -> bytes:
    # Encode as mesh with no faces (point cloud mode)
    return _encode_mesh_cli(positions, np.zeros((0, 3), dtype=np.uint32), qbits, 7)


def _decode_pc_cli(data: bytes) -> npt.NDArray:
    positions, _ = _decode_mesh_cli(data)
    return positions


def _write_temp_ply(path: Path, positions: npt.NDArray, faces: npt.NDArray) -> None:
    """Write a minimal binary PLY for Draco CLI input."""
    n_verts = len(positions)
    n_faces = len(faces)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_verts}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {n_faces}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(positions.astype(np.float32).tobytes())
        for face in faces:
            f.write(np.uint8(len(face)).tobytes())
            f.write(face.astype(np.int32).tobytes())


def _read_temp_ply(path: Path) -> tuple[npt.NDArray, npt.NDArray]:
    """Read a minimal PLY (from Draco CLI output)."""
    with open(path, "rb") as f:
        # Parse header
        n_verts = 0
        n_faces = 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
            elif line == "end_header":
                break

        positions = np.frombuffer(f.read(n_verts * 12), dtype=np.float32).reshape(-1, 3)

        faces_list: list[list[int]] = []
        for _ in range(n_faces):
            count = np.frombuffer(f.read(1), dtype=np.uint8)[0]
            face = np.frombuffer(f.read(count * 4), dtype=np.int32)
            faces_list.append(face.tolist())

    faces = np.array(faces_list, dtype=np.int64) if faces_list else np.zeros((0, 3), dtype=np.int64)
    return positions.copy(), faces
