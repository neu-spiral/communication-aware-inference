"""
Assemble the analysis root for the box-ray version of the quantization plots.

jensen_concavity.py / plot_concavity_per_method.py scan ONE tree for
``*_rays.json``. We want the topk and llmint8 arms exactly as they are, but the
quantization arm to come from the random-box-ray resampling
(outputs/quant_boxrays) instead of the axis-aligned enumeration. Mixing both in
one tree would double-count the quantization cell, so this builds a mirror of
real directories holding symlinks:

    non-quantization  -> the original mc_concavity ray/partial files
    quantization      -> the *_rays.json written by quant_box_rays.py

Nothing is moved or deleted; drop the mirror and the originals are untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent      # experiments/concavity
REPO_ROOT = HERE.parents[1]                 # repo root


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mc_root", default=str(REPO_ROOT / "outputs" / "mc_concavity"))
    p.add_argument("--box_root", default=str(REPO_ROOT / "outputs" / "quant_boxrays"))
    p.add_argument("--out", default=str(REPO_ROOT / "outputs" / "_analysis_boxrays"))
    args = p.parse_args()

    mc, box, out_top = Path(args.mc_root), Path(args.box_root), Path(args.out)
    if out_top.exists():
        for f in sorted(out_top.rglob("*"), reverse=True):
            f.unlink() if f.is_file() or f.is_symlink() else f.rmdir()
        out_top.rmdir()
    # jensen_concavity.py recovers model/dataset/metric for metadata-less
    # partial checkpoints by finding a path component literally named
    # "mc_concavity" (jensen_concavity.py:116). Keep that component or those
    # files collapse into a bogus "?" curve instead of joining their group.
    out = out_top / "mc_concavity"

    n_other = n_quant = 0
    # Everything that is NOT quantization, verbatim.
    for f in sorted(mc.rglob("*.json")):
        if not (f.name.endswith("_rays.json") or f.name.endswith(".partial.json")):
            continue
        if "quantization" in str(f.relative_to(mc)):
            continue
        dst = out / f.relative_to(mc)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(f.resolve())
        n_other += 1
    # Quantization comes from the box-ray resampling instead.
    for f in sorted(box.rglob("*_rays.json")):
        dst = out / f.relative_to(box)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(f.resolve())
        n_quant += 1

    print(f"linked {n_other} non-quantization ray files")
    print(f"linked {n_quant} quantization BOX-ray files")
    print(f"analysis root: {out_top}")


if __name__ == "__main__":
    main()
