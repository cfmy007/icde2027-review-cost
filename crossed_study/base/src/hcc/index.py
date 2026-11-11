"""Build reusable caches from CASIA-HWDB .gnt files in a ZIP archive.

The output NPZ contains normalized square grayscale images X, class IDs y,
decoded labels, remapped writer IDs, and class names. X uses uint8 with zero
as background; y and writer use int32.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np

from .gnt import iter_gnt_from_zip, to_square


def _writer_id(member: str) -> int:
    """Extract a writer ID from a .gnt archive member name."""
    stem = Path(member).stem
    m = re.search(r"\d+", stem)
    if m is None:
        raise ValueError(f"Cannot parse writer ID from filename: {member}")
    return int(m.group(0))


def list_members(zip_path: str | Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return sorted(n for n in zf.namelist() if n.lower().endswith(".gnt"))


def build_cache(
    zip_path: str | Path,
    out_path: str | Path,
    size: int = 32,
    limit_writers: int | None = None,
    only_chars: set[str] | None = None,
    verbose: bool = True,
) -> Path:
    """Parse .gnt archive members and write an NPZ cache.

    If only_chars is provided, skip records outside that set before resizing.
    """
    zip_path = Path(zip_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    only_tags: set[bytes] | None = None
    if only_chars is not None:
        only_tags = set()
        for ch in only_chars:
            try:
                only_tags.add(ch.encode("gbk"))
            except UnicodeEncodeError:
                continue

    members = list_members(zip_path)
    if limit_writers:
        members = members[:limit_writers]

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []

    with zipfile.ZipFile(zip_path) as zf:
        for i, name in enumerate(members, 1):
            writer = _writer_id(name)
            rows, wtags = [], []
            for rec in iter_gnt_from_zip(zf, name):
                if only_tags is not None and rec.tag not in only_tags:
                    continue
                rows.append((to_square(rec.ink, size=size) * 255).astype(np.uint8))
                wtags.append(rec.tag)
            if not rows:
                continue
            xs.append(np.stack(rows))
            ys.append(np.array(wtags, dtype=bytes))
            ws.append(np.full(len(rows), writer, dtype=np.int32))
            if verbose and (i % 20 == 0 or i == len(members)):
                print(f"  [{i}/{len(members)}] writer {writer}: {len(rows)} samples", flush=True)

    if not xs:
        raise RuntimeError("No samples were parsed; check only_chars")

    X = np.concatenate(xs, axis=0)
    y_raw = np.concatenate(ys, axis=0)
    writer = np.concatenate(ws, axis=0)

    # Remap writer IDs to consecutive integers from 0 to W-1.
    writers_sorted = sorted(np.unique(writer).tolist())
    w_remap = {w: i for i, w in enumerate(writers_sorted)}
    writer = np.array([w_remap[int(w)] for w in writer], dtype=np.int32)

    uniq = sorted({t for t in y_raw.tolist()})
    tag_to_id = {t: k for k, t in enumerate(uniq)}
    y = np.array([tag_to_id[t] for t in y_raw.tolist()], dtype=np.int32)
    names = np.array([_decode(t) for t in uniq], dtype=object)

    np.savez_compressed(
        out_path, X=X, y=y, label=names[y], writer=writer, names=names
    )
    if verbose:
        print(
            f"Wrote {out_path}: {len(X)} samples / {len(names)} classes / "
            f"{len(np.unique(writer))} writers, {X.nbytes / 1e6:.0f} MB uncompressed",
            flush=True,
        )
    return out_path


def _decode(tag: bytes) -> str:
    for enc in ("gbk", "gb2312"):
        try:
            return tag.decode(enc).replace("\x00", "")
        except UnicodeDecodeError:
            continue
    return "0x" + tag[::-1].hex()


def load_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}
