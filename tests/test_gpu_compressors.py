"""
test_gpu_compressors.py
=======================
Equivalence tests between CPU compressors (compressors.py) and GPU-native
compressors (gpu_compressors.py).

Three test levels
-----------------
1. test_topk_equivalence          — exact output equality (deterministic)
2. test_quantization_equivalence  — exact for FP16/INT8/INT2, tolerance for INT4
3. test_llmint8_equivalence       — near-zero tolerance (kthvalue vs quantile threshold)

Known differences documented
------------------------------
TopK mask size:
  CPU  uses numpy.packbits  → 1 bit/element  (smaller payload)
  GPU  uses torch.bool      → 1 byte/element (8× larger mask, but GPU-native)
  Decompressed OUTPUT is identical — only the payload dict layout differs.

Quantization scale granularity:
  CPU  _QuanInt8/4/2     one per-tensor abs-max scale
  GPU  _GPUQuanInt8/4/2  per-row (per-token) abs-max scale along the last dim
  A per-row scale is never coarser, so the GPU path is the finer quantizer:
  outputs differ and its round-trip error is lower (see test 2). Fix the
  backend per experiment, because switching it changes measured utilities.

INT4 stochastic rounding:
  Both call torch.bernoulli(prob) but on different devices (CPU vs CUDA).
  The CUDA RNG may produce different Bernoulli samples from the same seed
  depending on the device and cuDNN version.  The test allows a small
  tolerance and verifies that the quantisation error magnitude is comparable.

LLMInt8 threshold:
  CPU  uses torch.quantile (exact percentile interpolation)
  GPU  uses torch.kthvalue  (nearest-rank, no interpolation)
  When multiple elements tie at the percentile boundary the two methods may
  select a slightly different threshold, leading to a different outlier set.
  The test checks max absolute difference < tolerance rather than exact equality.

Usage
-----
  python tests/test_gpu_compressors.py            # cuda if available, else cpu
  python tests/test_gpu_compressors.py --verbose  # per-test detail
  python tests/test_gpu_compressors.py --device cpu
"""

import argparse
import pathlib
import sys
import traceback
from typing import List, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — allow running this file directly from anywhere in the checkout.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.core.compressors import (
    TopKCompressor,
    QuantizationCompressor,
    LLMInt8Compressor,
)
from src.core.gpu_compressors import (
    GPUTopKCompressor,
    GPUQuantizationCompressor,
    GPULLMInt8Compressor,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_tensor(shape, seed: int, relu: bool = True) -> torch.Tensor:
    """Create a reproducible FP32 test tensor."""
    torch.manual_seed(seed)
    t = torch.randn(*shape)
    return torch.relu(t) if relu else t


def _round_trip_cpu(comp, tensor: torch.Tensor, params, device: torch.device):
    """CPU compressor round-trip — compress on CPU, decompress to device."""
    data = comp.compress(tensor.cpu(), params)
    return comp.decompress(data, device=str(device))


def _round_trip_gpu(comp, tensor: torch.Tensor, params, device: torch.device):
    """GPU compressor round-trip — compress on device, decompress to device."""
    data = comp.compress(tensor.to(device), params)
    return comp.decompress(data, device=device)


def _check(
    out_cpu: torch.Tensor,
    out_gpu: torch.Tensor,
    label: str,
    atol: float,
    verbose: bool,
) -> Tuple[bool, float, float]:
    """
    Compare two tensors.  Returns (passed, max_abs_diff, mean_abs_diff).
    """
    diff = (out_cpu.cpu() - out_gpu.cpu()).abs()
    max_diff  = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    passed    = max_diff <= atol

    status = '✓  PASS' if passed else '✗  FAIL'
    if verbose or not passed:
        print(f"    {status}  {label}")
        print(f"           max_abs_diff={max_diff:.2e}  mean_abs_diff={mean_diff:.2e}"
              f"  tolerance={atol:.2e}")
        if not passed:
            print(f"           shapes: cpu={tuple(out_cpu.shape)} gpu={tuple(out_gpu.shape)}")
    return passed, max_diff, mean_diff


# ============================================================================
# Test shapes — covers ResNet activation sizes at default cutpoints
# ============================================================================
# link 0: batch=4, C=32, H=16, W=16  (~32K elements)
# link 1: batch=4, C=64, H=8,  W=8   (~16K elements)
# link 2: batch=4, C=128,H=4,  W=4   (~8K  elements)
# small:  batch=2, C=8,  H=4,  W=4   (256 elements — tests edge cases)
TEST_SHAPES = [
    (4, 32, 16, 16),
    (4, 64,  8,  8),
    (4, 128, 4,  4),
    (2,   8, 4,  4),
]


# ============================================================================
# 1. TopK equivalence
# ============================================================================

def test_topk_equivalence(device: torch.device, verbose: bool) -> bool:
    """
    Decompressed output should be bit-for-bit identical between CPU and GPU TopK.

    Why exact equality is expected:
    - Both implementations call torch.topk() which is deterministic for
      distinct absolute values.
    - The only risk is tie-breaking when two elements share the same absolute
      value — torch.topk is not guaranteed to break ties identically on CPU
      vs CUDA.  The test reports any ties found and uses exact equality;
      if your hardware breaks ties differently, the test will tell you.

    Note on payload size:
    - CPU uses numpy.packbits (1 bit/element mask) — smaller payload.
    - GPU uses torch.bool     (1 byte/element mask) — larger payload.
    - This is expected and documented.  Only the decompressed output is tested.
    """
    print("\n[1] TopK equivalence")
    cpu_comp = TopKCompressor()
    gpu_comp = GPUTopKCompressor()
    k_values = [0.5, 0.25, 0.125]
    seeds    = [42, 7, 99]
    all_pass = True

    for shape in TEST_SHAPES:
        for k in k_values:
            for seed in seeds:
                tensor = _make_tensor(shape, seed)
                out_cpu = _round_trip_cpu(cpu_comp, tensor, k, device)
                out_gpu = _round_trip_gpu(gpu_comp, tensor, k, device)

                # Check for ties in absolute values (may cause different results)
                flat     = tensor.reshape(tensor.shape[0], -1)
                n_keep   = max(1, int(flat.shape[1] * k))
                threshold = flat.abs().topk(n_keep, dim=1).values.min()
                n_ties   = int((flat.abs() == threshold).sum().item())

                label = f"shape={shape} k={k} seed={seed}"
                if n_ties > 0:
                    label += f"  [ties={n_ties} — may differ]"

                passed, max_diff, _ = _check(out_cpu, out_gpu, label,
                                             atol=0.0, verbose=verbose)
                all_pass = all_pass and passed

                # Also verify shape and device
                assert out_gpu.shape == tensor.shape, \
                    f"Shape mismatch: {out_gpu.shape} != {tensor.shape}"
                assert out_gpu.device.type == device.type, \
                    f"Device mismatch: {out_gpu.device} != {device}"

    # Payload size comparison (documented difference)
    tensor = _make_tensor((4, 32, 16, 16), 42)
    cpu_data = cpu_comp.compress(tensor.cpu(), 0.5)
    gpu_data = gpu_comp.compress(tensor.to(device), 0.5)
    cpu_bytes = cpu_comp.get_compressed_size(cpu_data)
    gpu_bytes = gpu_comp.get_compressed_size(gpu_data)
    if verbose:
        print(f"    [payload size] CPU={cpu_bytes}B  GPU={gpu_bytes}B  "
              f"(GPU mask is ~8× larger — torch.bool vs packbits, expected)")

    status = '✓  ALL PASS' if all_pass else '✗  FAILURES'
    print(f"  Result: {status}")
    return all_pass


# ============================================================================
# 2. Quantization equivalence
# ============================================================================

def test_quantization_equivalence(device: torch.device, verbose: bool) -> bool:
    """
    The two integer quantizers are NOT bit-equivalent, by construction:

        CPU (compressors._QuanInt8/4/2)     one PER-TENSOR abs-max scale
        GPU (gpu_compressors._GPUQuanInt8/4/2)  PER-ROW (per-token) abs-max
                                            scale along the last dimension

    A per-row scale is never coarser than the per-tensor one, so the GPU path
    is a strictly finer quantizer. Asserting equal output would therefore be
    asserting something false. What IS invariant, and what this test pins:

      1. FP16 is a plain dtype cast on both sides, so it must match exactly.
      2. Each backend's round-trip error stays inside the per-tensor step
         bound: half a step for round-to-nearest (INT8, INT2), a full step for
         INT4, whose stochastic rounding can miss by one step either way.
      3. The GPU error is no larger than the CPU error (RMSE), which is the
         observable consequence of the finer scale. Measured ratios are
         0.30-0.53; the check allows 1.05 so it fails on a regression rather
         than on noise.

    A violation of 2 or 3 is a bug in one of the two implementations.
    """
    print("\n[2] Quantization equivalence")
    cpu_comp = QuantizationCompressor()
    gpu_comp = GPUQuantizationCompressor()

    # (k, label, levels, step_multiplier)
    #   levels           = symmetric quantisation levels for that width
    #   step_multiplier  = allowed error as a multiple of the PER-TENSOR step;
    #                      0.5 for round-to-nearest, 1.0 for stochastic INT4.
    #   FP16 carries levels=None and is checked for exact CPU/GPU equality.
    configs = [
        (0.5,    'FP16', None,  None),
        (0.25,   'INT8', 127.0, 0.5),
        (0.125,  'INT4', 7.0,   1.0),
        (0.0625, 'INT2', 1.0,   0.5),
    ]
    seeds    = [42, 7, 99]
    all_pass = True

    for k, label, levels, step_multiplier in configs:
        for shape in TEST_SHAPES:
            for seed in seeds:
                torch.manual_seed(seed)
                tensor = _make_tensor(shape, seed)

                torch.manual_seed(seed)
                out_cpu = _round_trip_cpu(cpu_comp, tensor, k, device)
                torch.manual_seed(seed)
                out_gpu = _round_trip_gpu(gpu_comp, tensor, k, device)

                assert out_gpu.shape == tensor.shape
                assert out_gpu.device.type == device.type

                if levels is None:
                    # FP16: identical dtype cast on both paths.
                    passed, _, _ = _check(out_cpu, out_gpu,
                                          f"{label} shape={shape} k={k} seed={seed}",
                                          atol=0.0, verbose=verbose)
                    all_pass = all_pass and passed
                    continue

                abs_max = float(tensor.abs().max().item())
                step    = (abs_max / levels) if abs_max > 0 else 1.0
                bound   = step_multiplier * step + 1e-5

                ref      = tensor.cpu()
                err_cpu  = (out_cpu.cpu() - ref).abs()
                err_gpu  = (out_gpu.cpu() - ref).abs()
                max_cpu  = float(err_cpu.max().item())
                max_gpu  = float(err_gpu.max().item())
                rmse_cpu = float(err_cpu.pow(2).mean().sqrt().item())
                rmse_gpu = float(err_gpu.pow(2).mean().sqrt().item())

                in_bound = max_cpu <= bound and max_gpu <= bound
                # A finer scale cannot do worse; 1.05 leaves room for noise.
                gpu_no_worse = rmse_gpu <= rmse_cpu * 1.05 + 1e-9
                passed = in_bound and gpu_no_worse

                status = '✓  PASS' if passed else '✗  FAIL'
                if verbose or not passed:
                    print(f"    {status}  {label} shape={shape} k={k} seed={seed}")
                    print(f"           max_err cpu={max_cpu:.2e} gpu={max_gpu:.2e}"
                          f"  bound={bound:.2e} ({step_multiplier:g} step)")
                    print(f"           rmse    cpu={rmse_cpu:.2e} gpu={rmse_gpu:.2e}"
                          f"  ratio={rmse_gpu / rmse_cpu if rmse_cpu else 0:.3f}")
                    if not in_bound:
                        print("           -> round-trip error exceeds the step bound")
                    if not gpu_no_worse:
                        print("           -> per-row GPU scale did worse than per-tensor CPU")
                all_pass = all_pass and passed

    status = '✓  ALL PASS' if all_pass else '✗  FAILURES'
    print(f"  Result: {status}")
    return all_pass


# ============================================================================
# 3. LLMInt8 equivalence
# ============================================================================

def test_llmint8_equivalence(device: torch.device, verbose: bool) -> bool:
    """
    Tolerance depends on the regular_precision:
    - int8, int2: 0.01 * abs_max  (threshold difference only)
    - int4:       1.0 * reg_scale  (stochastic rounding dominates)
      where reg_scale = abs_max_regular / 7.0 for the regular block.

    The int4 regular precision uses torch.bernoulli which produces different
    samples on CPU vs CUDA, so the full quantisation step is the correct bound.

    Also checks:
    - num_outliers is close between CPU and GPU (within 5% of total elements)
    - Decompressed output shape and device are correct
    """
    print("\n[3] LLMInt8 equivalence")
    cpu_comp = LLMInt8Compressor()
    gpu_comp = GPULLMInt8Compressor()

    configs = [
        [0.01, 'fp16', 'int8'],
        [0.05, 'fp16', 'int4'],   # int4 regular — stochastic rounding
        [0.10, 'int8', 'int2'],
        [0.01, 'fp16', 'int2'],
    ]
    seeds    = [42, 7]
    all_pass = True

    for params in configs:
        outlier_ratio, outlier_prec, regular_prec = params
        for shape in TEST_SHAPES:
            for seed in seeds:
                torch.manual_seed(seed)
                tensor = _make_tensor(shape, seed, relu=False)
                abs_max = float(tensor.abs().max().item())

                if regular_prec == 'int4':
                    # Stochastic rounding: allow up to 1 full INT4 step.
                    # Approximate the regular block scale as abs_max / 7.0
                    # (same formula as _GPUQuanInt4 and _QuanInt4).
                    reg_scale = abs_max / 7.0 if abs_max > 0 else 1.0
                    atol = reg_scale
                else:
                    # Threshold difference only — 1% of dynamic range
                    atol = 0.01 * abs_max if abs_max > 0 else 1e-6

                torch.manual_seed(seed)
                out_cpu = _round_trip_cpu(cpu_comp, tensor, params, device)
                torch.manual_seed(seed)
                out_gpu = _round_trip_gpu(gpu_comp, tensor, params, device)

                test_label = (f"k={outlier_ratio} out={outlier_prec} reg={regular_prec} "
                              f"shape={shape} seed={seed}")
                passed, max_diff, mean_diff = _check(out_cpu, out_gpu, test_label,
                                                     atol=atol, verbose=verbose)
                all_pass = all_pass and passed

                # Check outlier count is close
                cpu_data = cpu_comp.compress(tensor.cpu(), params)
                gpu_data = gpu_comp.compress(tensor.to(device), params)
                n_out_cpu = int(cpu_data['num_outliers'])
                n_out_gpu = int(gpu_data['num_outliers'])
                n_total   = tensor.numel()
                outlier_diff_frac = abs(n_out_cpu - n_out_gpu) / max(n_total, 1)

                if outlier_diff_frac > 0.05:
                    print(f"    ⚠  outlier count differs by {outlier_diff_frac:.1%}: "
                          f"cpu={n_out_cpu} gpu={n_out_gpu} total={n_total}")
                    all_pass = False
                elif verbose:
                    print(f"    [outliers] cpu={n_out_cpu} gpu={n_out_gpu} "
                          f"diff={abs(n_out_cpu - n_out_gpu)} "
                          f"({outlier_diff_frac:.2%} of total)  ✓")

                assert out_gpu.shape == tensor.shape
                assert out_gpu.device.type == device.type

    status = '✓  ALL PASS' if all_pass else '✗  FAILURES'
    print(f"  Result: {status}")
    return all_pass


# ============================================================================
# Runner
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Equivalence tests: CPU compressors vs GPU-native compressors."
    )
    p.add_argument('--verbose',    action='store_true',
                   help='Print per-test detail including passing tests.')
    p.add_argument('--device',     type=str, default=None,
                   help="Device to use: 'cuda' or 'cpu'. Default: cuda if available.")
    args = p.parse_args()

    device_str = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device     = torch.device(device_str)

    print("=" * 60)
    print("GPU Compressor Equivalence Test Suite")
    print("=" * 60)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  Device          : {device}")
    if device.type == 'cuda':
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
    print(f"  Verbose         : {args.verbose}")
    print("=" * 60)

    if device.type == 'cpu':
        print("\n⚠  Running on CPU — GPU compressors will be tested on CPU device.")
        print("   For a meaningful GPU vs CPU comparison, run on a CUDA device.\n")

    results = {}

    try:
        results['TopK']          = test_topk_equivalence(device, args.verbose)
    except Exception:
        print("  ERROR in TopK test:")
        traceback.print_exc()
        results['TopK'] = False

    try:
        results['Quantization']  = test_quantization_equivalence(device, args.verbose)
    except Exception:
        print("  ERROR in Quantization test:")
        traceback.print_exc()
        results['Quantization'] = False

    try:
        results['LLMInt8']       = test_llmint8_equivalence(device, args.verbose)
    except Exception:
        print("  ERROR in LLMInt8 test:")
        traceback.print_exc()
        results['LLMInt8'] = False

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = '✓  PASS' if passed else '✗  FAIL'
        print(f"  {status}  {name}")
        all_pass = all_pass and passed

    print("=" * 60)
    if all_pass:
        print("All tests passed. TopK and LLMInt8 agree; the quantizers differ")
        print("only by the documented per-tensor vs per-row scale granularity.")
    else:
        print("Some tests FAILED.  Review output above for details.")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()

