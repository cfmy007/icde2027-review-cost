"""Read and write CASIA-HWDB .gnt records.

The format is described in the official Offline_database.html, Table 2:

    Sample size  uint32   4 bytes  Total record length, including the header
    Tag code     char[2]  2 bytes  Character label in GB encoding
    Width        uint16   2 bytes
    Height       uint16   2 bytes
    Bitmap       uint8    W * H   Row-major pixels; background 255

Foreground pixels have values below 255.
"""

from __future__ import annotations

import io
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

HEADER = 10  # 4 + 2 + 2 + 2
_BIN_THRESHOLD = 250  # Grayscale values below this threshold are foreground.


@dataclass
class Record:
    """One character sample."""

    tag: bytes  # Two-byte GB-encoded label.
    width: int
    height: int
    bitmap: np.ndarray  # (height, width) uint8; background 255, foreground 0..254.

    @property
    def label(self) -> str:
        """Decode the label as Unicode, removing null bytes from ASCII tags.
        """
        for enc in ("gbk", "gb2312"):
            try:
                return self.tag.decode(enc).replace("\x00", "")
            except UnicodeDecodeError:
                continue
        return "0x" + self.tag[::-1].hex()

    @property
    def ink(self) -> np.ndarray:
        """Return a float32 foreground intensity image in [0, 1]."""
        return (255.0 - self.bitmap.astype(np.float32)) / 255.0


def iter_records(data: bytes) -> Iterator[Record]:
    """Iterate over records in an in-memory .gnt file."""
    offset = 0
    n = len(data)
    while offset + HEADER <= n:
        (size,) = struct.unpack_from("<I", data, offset)
        if size < HEADER or offset + size > n:
            raise ValueError(f"Corrupt record: offset={offset} size={size} file_len={n}")
        tag = data[offset + 4 : offset + 6]
        width, height = struct.unpack_from("<HH", data, offset + 6)
        bitmap = np.frombuffer(
            data, dtype=np.uint8, count=width * height, offset=offset + HEADER
        ).reshape(height, width)
        yield Record(tag=tag, width=width, height=height, bitmap=bitmap)
        offset += size


def read_gnt(path: str | Path) -> list[Record]:
    return list(iter_records(Path(path).read_bytes()))


def iter_gnt_from_zip(zf: zipfile.ZipFile, name: str) -> Iterator[Record]:
    """Read and parse a .gnt member without extracting the archive."""
    with zf.open(name) as fh:
        yield from iter_records(fh.read())


def bounding_box(ink: np.ndarray, threshold: float = 0.02) -> tuple[int, int, int, int]:
    """Return the inclusive foreground bounding box (top, bottom, left, right)."""
    mask = ink > threshold
    if not mask.any():
        return 0, ink.shape[0] - 1, 0, ink.shape[1] - 1
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def to_square(ink: np.ndarray, size: int = 32, pad_ratio: float = 0.10) -> np.ndarray:
    """Crop to the foreground and resize into a centered, zero-padded square.

    The preprocessing does not deskew or apply morphological thinning, retaining
    writer-style information for the clustering analysis.
    """
    t, b, l, r = bounding_box(ink)
    crop = ink[t : b + 1, l : r + 1]
    h, w = crop.shape
    inner = max(1, int(round(size * (1.0 - 2 * pad_ratio))))
    scale = inner / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # Resize by nearest-neighbour indexing with NumPy.
    ys = (np.arange(new_h) * (h / new_h)).astype(np.int32)
    xs = (np.arange(new_w) * (w / new_w)).astype(np.int32)
    small = crop[np.clip(ys, 0, h - 1)][:, np.clip(xs, 0, w - 1)]

    out = np.zeros((size, size), dtype=np.float32)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    out[top : top + new_h, left : left + new_w] = small
    return out


def synthetic_gnt(records: list[tuple[bytes, np.ndarray]]) -> bytes:
    """Build a .gnt byte stream for unit tests."""
    buf = io.BytesIO()
    for tag, bitmap in records:
        h, w = bitmap.shape
        size = HEADER + w * h
        buf.write(struct.pack("<I", size))
        buf.write(tag)
        buf.write(struct.pack("<HH", w, h))
        buf.write(bitmap.astype(np.uint8).tobytes())
    return buf.getvalue()
