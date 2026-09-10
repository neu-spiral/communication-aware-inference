"""
gpu_compressors.py  —  GPU-Native Activation Compressor Library
===============================================================

Drop-in GPU counterparts for every compressor in compressors.py.
All operations stay on the input tensor's device (CUDA or CPU) throughout
compress() and decompress() — no .cpu() calls, no numpy, no packbits.

The compressed payload is a dict of torch.Tensors that live on the same
device as the input.  This eliminates:
  • PCIe round-trips (GPU→CPU→GPU) on every accuracy evaluation
  • numpy dtype conversions
  • torch.quantile() CPU fallback (replaced with torch.kthvalue / argsort)

API is identical to compressors.py so they are interchangeable:

    compressed = compressor.compress(tensor, compression_params)
    restored   = compressor.decompress(compressed, device='cuda')
    nbytes     = compressor.get_compressed_size(compressed)
    name       = compressor.get_name()

Quick-start
-----------
    from gpu_compressors import (
        GPUTopKCompressor, GPUQuantizationCompressor, GPULLMInt8Compressor,
        get_gpu_compressor,
    )

    comp = get_gpu_compressor('topk')
    data = comp.compress(tensor.cuda(), 0.5)    # stays on GPU
    out  = comp.decompress(data, device='cuda') # stays on GPU

Key differences from compressors.py
------------------------------------
  1. No numpy dependency — bit-packing is done with PyTorch bitwise ops.
  2. No .cpu() transfer inside compress() or decompress().
  3. Payload tensors are stored on the input device; decompress() moves
     them to the requested device with a single .to(device) if needed.
  4. LLMInt8: torch.quantile() replaced with GPU-native kthvalue which
     avoids the expensive CPU fallback for large tensors.
  5. get_compressed_size() estimates bytes from tensor element counts
     (no numpy .nbytes — payload stays as tensors).

Dependencies: torch  (numpy not required)
"""

import torch


# ============================================================================
# Base class
# ============================================================================

class GPUBaseCompressor:
    """
    Abstract base for GPU-native compressors.
    Identical interface to BaseCompressor in compressors.py.
    """

    def compress(self, tensor: torch.Tensor, compression_params) -> dict:
        raise NotImplementedError

    def decompress(self, compressed: dict, device=None) -> torch.Tensor:
        raise NotImplementedError

    def get_compressed_size(self, compressed: dict) -> int:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError


# ============================================================================
# GPU TopK Compressor
# ============================================================================

class GPUTopKCompressor(GPUBaseCompressor):
    """
    GPU-native Top-K sparsification compressor.

    Differences from TopKCompressor (compressors.py)
    -------------------------------------------------
    • Mask stored as a torch.bool tensor on the GPU (not numpy packbits).
      Boolean tensors use 1 byte/element in PyTorch storage — 8× less
      efficient than packbits, but avoids any CPU transfer.
    • Values stored as a float tensor (not numpy array).
    • No .cpu(), no numpy.packbits / unpackbits.

    Compression ratio (payload / original):
      k (values) + 1/4 (bool mask at 1B/elem vs 4B/elem FP32) ≈ k + 0.25
      e.g. k=0.5 → ~75% of original.  Slightly worse than packbits
      (which achieves k + 1/8) but GPU-native with no transfer cost.

    Parameters
    ----------
    compression_params : float
        Fraction of elements to keep (0 < k ≤ 1).
    """

    def compress(self, tensor: torch.Tensor, compression_params) -> dict:
        k_ratio = float(compression_params)
        device  = tensor.device
        original_shape = tensor.shape
        batch_size = original_shape[0]

        # Flatten to [batch, N]
        reshaped = tensor.reshape(batch_size, -1)
        elements_per_sample = reshaped.shape[1]
        k = max(1, int(elements_per_sample * k_ratio))

        # Per-sample top-K indices by absolute value
        _, top_indices = torch.topk(reshaped.abs(), k, dim=1)

        # Boolean mask [batch, N] — stays on device
        mask = torch.zeros(batch_size, elements_per_sample,
                           dtype=torch.bool, device=device)
        mask.scatter_(1, top_indices, True)

        # Values in positional order
        sorted_indices = torch.sort(top_indices, dim=1)[0]
        sorted_values  = torch.gather(reshaped, 1, sorted_indices)  # [batch, k]

        return {
            'method':              'gpu_topk',
            'values':              sorted_values,          # [batch, k] float on device
            'mask':                mask,                   # [batch, N] bool on device
            'shape':               original_shape,
            'elements_per_sample': elements_per_sample,
            'k':                   k,
            'device':              device,
        }

    def decompress(self, compressed: dict, device=None) -> torch.Tensor:
        tgt = device or compressed['device']
        values = compressed['values'].to(tgt)
        mask   = compressed['mask'].to(tgt)
        shape  = compressed['shape']
        batch_size = shape[0]
        elements_per_sample = compressed['elements_per_sample']

        reshaped = torch.zeros(batch_size, elements_per_sample,
                               dtype=values.dtype, device=tgt)
        reshaped[mask] = values.flatten()
        return reshaped.reshape(shape)

    def get_compressed_size(self, compressed: dict) -> int:
        # values: float32 (4 bytes/elem), mask: bool (1 byte/elem)
        values_bytes = compressed['values'].numel() * 4
        mask_bytes   = compressed['mask'].numel() * 1
        return values_bytes + mask_bytes + 16  # +16 metadata

    def get_name(self) -> str:
        return "GPUTopK"


# ============================================================================
# GPU Quantization sub-compressors
# ============================================================================

_FP16_MAX_FINITE = 65504.0  # largest finite magnitude representable in IEEE FP16


class _GPUQuanFP16:
    """FP32 → FP16 → FP32.  Everything on device, no numpy."""

    def compress(self, tensor: torch.Tensor, _params) -> dict:
        return {
            'method': 'gpu_quan_fp16',
            # Paper spec: direct datatype conversion *with value clipping* --
            # a bare .half() sends anything above the FP16 range to +/-inf.
            'values': tensor.clamp(-_FP16_MAX_FINITE, _FP16_MAX_FINITE).half(),
            'shape':  tensor.shape,
            'device': tensor.device,
        }

    def decompress(self, data: dict, device=None) -> torch.Tensor:
        tgt = device or data['device']
        return data['values'].to(tgt).float().reshape(data['shape'])

    def get_compressed_size(self, data: dict) -> int:
        return data['values'].numel() * 2 + 16  # FP16 = 2 bytes


class _GPUQuanInt8:
    """Symmetric per-token/per-row INT8 quantization (scale along last dim)."""

    def compress(self, tensor: torch.Tensor, _params) -> dict:
        cols = int(tensor.shape[-1])
        x2 = tensor.reshape(-1, cols).float()                 # [rows, cols]
        row_scale = (x2.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
        quantized = (x2 / row_scale).round().clamp(-127, 127).to(torch.int8)
        return {
            'method':    'gpu_quan_int8',
            'values':    quantized,       # int8 [rows, cols] on device
            'row_scale': row_scale,       # [rows, 1] on device
            'cols':      cols,
            'shape':     tensor.shape,
            'device':    tensor.device,
        }

    def decompress(self, data: dict, device=None) -> torch.Tensor:
        tgt = device or data['device']
        vals = data['values'].to(tgt).float()                 # [rows, cols]
        return (vals * data['row_scale'].to(tgt)).reshape(data['shape'])

    def get_compressed_size(self, data: dict) -> int:
        # INT8 values + per-row scales (4B each) + metadata.
        return data['values'].numel() * 1 + data['row_scale'].numel() * 4 + 16


class _GPUQuanInt4:
    """
    4-bit quantization with stochastic rounding.
    Bit-packing done with PyTorch bitwise ops — no numpy.
    Two 4-bit values packed into one uint8 byte.
    """

    def compress(self, tensor: torch.Tensor, _params) -> dict:
        device  = tensor.device
        shape   = tensor.shape
        cols    = int(shape[-1])
        x2 = tensor.reshape(-1, cols).float()                 # [rows, cols]
        row_scale = (x2.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-8)

        # Per-row scale, stochastic rounding on device.
        x       = x2 / row_scale
        floor_x = x.floor()
        prob    = (x - floor_x).clamp(0.0, 1.0)
        quantized = (floor_x + torch.bernoulli(prob)).clamp(-7, 7).to(torch.int8)
        shifted   = (quantized + 7).to(torch.uint8)   # [0, 14]

        flat = shifted.flatten()                       # row-major (row 0 first)
        n    = flat.numel()
        padding = (2 - n % 2) % 2
        if padding:
            flat = torch.cat([flat,
                              torch.zeros(padding, dtype=torch.uint8, device=device)])

        # Pack: high nibble | low nibble
        flat2  = flat.reshape(-1, 2)
        packed = ((flat2[:, 0] << 4) | (flat2[:, 1] & 0x0F)).to(torch.uint8)

        return {
            'method':    'gpu_quan_int4',
            'packed':    packed,          # uint8 on device
            'row_scale': row_scale,       # [rows, 1]
            'cols':      cols,
            'shape':     shape,
            'n_elements': n,
            'padding':   padding,
            'device':    device,
        }

    def decompress(self, data: dict, device=None) -> torch.Tensor:
        tgt    = device or data['device']
        packed = data['packed'].to(tgt)
        high   = (packed >> 4) & 0x0F
        low    =  packed        & 0x0F
        unpacked = torch.stack([high, low], dim=1).flatten()
        if data['padding']:
            unpacked = unpacked[:data['n_elements']]
        quantized = (unpacked.to(torch.int8) - 7).float().reshape(-1, data['cols'])
        return (quantized * data['row_scale'].to(tgt)).reshape(data['shape'])

    def get_compressed_size(self, data: dict) -> int:
        return data['packed'].numel() + 4 + 16  # packed bytes + scale + meta


class _GPUQuanInt2:
    """
    2-bit quantization.
    Four 2-bit values packed into one uint8 byte — GPU-native.
    """

    def compress(self, tensor: torch.Tensor, _params) -> dict:
        device  = tensor.device
        shape   = tensor.shape
        cols    = int(shape[-1])
        x2 = tensor.reshape(-1, cols).float()                 # [rows, cols]
        row_scale = x2.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # INT2 qmax=1

        quantized = (x2 / row_scale).round().clamp(-1, 1).to(torch.int8)
        shifted   = (quantized + 1).to(torch.uint8)   # {0, 1, 2}

        flat = shifted.flatten()                       # row-major
        n    = flat.numel()
        padding = (4 - n % 4) % 4
        if padding:
            flat = torch.cat([flat,
                              torch.zeros(padding, dtype=torch.uint8, device=device)])

        # Pack: 4 x 2-bit → 1 x uint8
        flat4  = flat.reshape(-1, 4)
        packed = (  (flat4[:, 0] << 6)
                  | (flat4[:, 1] << 4)
                  | (flat4[:, 2] << 2)
                  |  flat4[:, 3]       ).to(torch.uint8)

        return {
            'method':    'gpu_quan_int2',
            'packed':    packed,
            'row_scale': row_scale,       # [rows, 1]
            'cols':      cols,
            'shape':     shape,
            'n_elements': n,
            'padding':   padding,
            'device':    device,
        }

    def decompress(self, data: dict, device=None) -> torch.Tensor:
        tgt    = device or data['device']
        packed = data['packed'].to(tgt)
        b0 = (packed >> 6) & 0x03
        b1 = (packed >> 4) & 0x03
        b2 = (packed >> 2) & 0x03
        b3 =  packed        & 0x03
        unpacked = torch.stack([b0, b1, b2, b3], dim=1).flatten()
        if data['padding']:
            unpacked = unpacked[:data['n_elements']]
        quantized = (unpacked.to(torch.int8) - 1).float().reshape(-1, data['cols'])
        return (quantized * data['row_scale'].to(tgt)).reshape(data['shape'])

    def get_compressed_size(self, data: dict) -> int:
        return data['packed'].numel() + 4 + 16


# ============================================================================
# GPU Quantization Compressor (dispatcher)
# ============================================================================

class GPUQuantizationCompressor(GPUBaseCompressor):
    """
    GPU-native quantization dispatcher — mirrors QuantizationCompressor.

    k → bit-width mapping (identical to compressors.py):
        0.5    → FP16   (2×)
        0.25   → INT8   (4×)
        0.125  → INT4   (8×)
        0.0625 → INT2   (16×)
        other  → FP16   (fallback)

    All sub-quantizers operate entirely on the tensor's device.
    """

    def __init__(self):
        self._fp16 = _GPUQuanFP16()
        self._int8 = _GPUQuanInt8()
        self._int4 = _GPUQuanInt4()
        self._int2 = _GPUQuanInt2()

    def _select(self, k: float):
        if   k < 0.0625+ 1e-6: return self._int2
        elif k < 0.125+1e-6: return self._int4
        elif k < 0.25+1e-6: return self._int8
        elif k < 0.5+1e-6: return self._fp16
        else:                         return self._fp16

    def _dispatch(self, data: dict):
        m = data.get('method', '')
        if   m == 'gpu_quan_int2': return self._int2
        elif m == 'gpu_quan_int4': return self._int4
        elif m == 'gpu_quan_int8': return self._int8
        else:                      return self._fp16

    def compress(self, tensor: torch.Tensor, compression_params) -> dict:
        #TT
        return self._select(float(compression_params)).compress(tensor, compression_params)

    def decompress(self, compressed: dict, device=None) -> torch.Tensor:
        return self._dispatch(compressed).decompress(compressed, device)

    def get_compressed_size(self, compressed: dict) -> int:
        return self._dispatch(compressed).get_compressed_size(compressed)

    def get_name(self) -> str:
        return "GPUQuantization"


# ============================================================================
# GPU LLMInt8 Compressor
# ============================================================================

class GPULLMInt8Compressor(GPUBaseCompressor):
    """
    GPU-native LLM.int8()-style hybrid mixed-precision compressor.

    Differences from LLMInt8Compressor (compressors.py)
    ----------------------------------------------------
    1. No torch.quantile() — replaced with GPU-native kthvalue().
       torch.quantile() silently falls back to CPU for large tensors
       (numel > 2^24), causing a PCIe round-trip on every eval.
       torch.kthvalue() stays on GPU regardless of tensor size.

    2. No numpy.packbits — bitmask stored as torch.bool on GPU.

    3. All intermediate tensors stay on the input device.
       decompress() moves tensors to the target device with a single
       .to(device) call per tensor, not per-element.

    4. Payload stores torch.Tensors, not numpy arrays.

    compression_params : list  [k, outlier_precision, regular_precision]
        k                  : float  outlier ratio (e.g. 0.01 = top 1%)
        outlier_precision  : str    'fp16' or 'int8'
        regular_precision  : str    'int8', 'int4', or 'int2'
    """

    # Sub-quantizers for regular values — reuse GPU quant classes
    def __init__(self):
        self._int8 = _GPUQuanInt8()
        self._int4 = _GPUQuanInt4()
        self._int2 = _GPUQuanInt2()

    def _regular_compressor(self, precision: str):
        if   precision == 'int4': return self._int4
        elif precision == 'int2': return self._int2
        else:                     return self._int8  # default / 'int8'

    # ------------------------------------------------------------------
    # GPU-native quantile via kthvalue
    # ------------------------------------------------------------------
    @staticmethod
    def _gpu_quantile(flat_abs: torch.Tensor, q: float) -> torch.Tensor:
        """
        Compute the q-th quantile of a 1-D tensor entirely on GPU.

        torch.quantile falls back to CPU for numel > 2^24 (~16 M elements).
        torch.kthvalue always stays on the input device.

        q=0.99 means 'the value below which 99% of elements fall',
        i.e. the threshold above which the top 1% outliers lie.
        """
        n = flat_abs.numel()
        if n == 0:
            return torch.tensor(0.0, device=flat_abs.device)
        # kthvalue is 1-indexed; clamp k to valid range
        k = max(1, min(n, int(q * n)))
        return torch.kthvalue(flat_abs, k).values

    # ------------------------------------------------------------------
    # compress
    # ------------------------------------------------------------------
    def compress(self, tensor: torch.Tensor, compression_params) -> dict:
        outlier_ratio, outlier_precision, regular_precision = compression_params
        device = tensor.device

        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        shape          = tensor.shape
        original_bytes = tensor.numel() * 4

        # ---- 1. Outlier threshold (GPU-native) ----
        flat_abs   = tensor.abs().reshape(-1)
        percentile = 1.0 - float(outlier_ratio)
        threshold  = self._gpu_quantile(flat_abs, percentile)

        mask_bool  = tensor.abs() > threshold          # bool, on device
        num_outliers = int(mask_bool.sum().item())

        # ---- 2. Extract and quantize outlier values ----
        outlier_raw = torch.masked_select(tensor, mask_bool)

        if outlier_precision == 'fp16':
            outlier_values = outlier_raw.half()        # on device
            size_outliers  = outlier_values.numel() * 2
            outlier_scale  = None
        else:  # 'int8'
            abs_max = outlier_raw.abs().max().clamp(min=1e-8)
            outlier_scale  = abs_max / 127.0
            outlier_values = (outlier_raw / outlier_scale).round().clamp(-127, 127).to(torch.int8)
            size_outliers  = outlier_values.numel()

        # ---- 3. Quantize regular values (row-wise AbsMax, GPU) ----
        tensor_reg = tensor.clone()
        tensor_reg.masked_fill_(mask_bool, 0.0)

        # reshape to [rows, hidden] for row-wise scaling
        flattened = tensor_reg.view(-1, shape[-1])
        row_scales = flattened.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)

        # Compress the regular block using the appropriate sub-quantizer
        # We pass the normalised matrix to the sub-quantizer
        reg_normalised = flattened / row_scales       # values in [-1, 1] roughly

        # Use simple per-row INT8/INT4/INT2 without stochastic rounding
        # by calling the sub-quantizer on the normalised block directly
        reg_comp  = self._regular_compressor(regular_precision)
        reg_data  = reg_comp.compress(reg_normalised, None)

        # ---- 4. Compute compressed size ----
        size_mask    = mask_bool.numel() * 1      # bool = 1 byte/elem
        size_scales  = row_scales.numel() * 4     # float32
        size_regular = reg_comp.get_compressed_size(reg_data)
        metadata_bytes = 64
        compressed_bytes = (size_mask + size_outliers + size_regular
                            + size_scales + metadata_bytes)
        compression_ratio = compressed_bytes / original_bytes if original_bytes else 1.0

        return {
            'method':            'gpu_llmint8_hybrid',
            'mask_bool':         mask_bool,           # [*shape] bool on device
            'outlier_values':    outlier_values,       # on device
            'outlier_scale':     outlier_scale,        # scalar tensor or None
            'reg_data':          reg_data,             # sub-compressor payload
            'row_scales':        row_scales,           # [rows, 1] float on device
            'shape':             shape,
            'numel':             tensor.numel(),
            'num_outliers':      num_outliers,
            'threshold':         threshold.item(),
            'outlier_precision': outlier_precision,
            'regular_precision': regular_precision,
            'original_bytes':    original_bytes,
            'compressed_bytes':  compressed_bytes,
            'compression_ratio': compression_ratio,
            'compressed':        True,
            'device':            device,
        }

    # ------------------------------------------------------------------
    # decompress
    # ------------------------------------------------------------------
    def decompress(self, data: dict, device=None) -> torch.Tensor:
        if not data.get('compressed', False):
            return data['tensor'].to(device or data['device'])

        tgt    = device or data['device']
        shape  = data['shape']

        # ---- 1. Dequantize regular values ----
        reg_prec  = data['regular_precision']
        reg_comp  = self._regular_compressor(reg_prec)
        row_scales = data['row_scales'].to(tgt)

        # decompress the normalised block
        reg_norm = reg_comp.decompress(data['reg_data'], device=tgt)  # [rows, hidden]
        restored = (reg_norm * row_scales).view(shape)

        # ---- 2. Scatter outliers back ----
        mask_bool = data['mask_bool'].to(tgt)
        if data['num_outliers'] > 0:
            o_prec = data['outlier_precision']
            if o_prec == 'fp16':
                outlier_vals = data['outlier_values'].to(tgt).float()
            else:  # int8
                outlier_vals = (data['outlier_values'].to(tgt).float()
                                * data['outlier_scale'].to(tgt))
            restored.masked_scatter_(mask_bool, outlier_vals)

        return restored

    def get_compressed_size(self, compressed: dict) -> int:
        if not compressed.get('compressed', False):
            return compressed.get('original_bytes', 0)
        return int(compressed['compressed_bytes'])

    def get_name(self) -> str:
        return "GPULLMInt8"


# ============================================================================
# Convenience factory
# ============================================================================

def get_gpu_compressor(name: str) -> GPUBaseCompressor:
    """
    Factory function — mirrors get_compressor() in compressors.py.

    Args:
        name: One of 'topk', 'quantization' / 'quant', 'llmint8'.

    Returns:
        GPU-native compressor instance.

    Example:
        comp = get_gpu_compressor('topk')
        data = comp.compress(tensor.cuda(), 0.5)
        out  = comp.decompress(data, device='cuda')
    """
    n = name.lower().strip()
    if n == 'topk':
        return GPUTopKCompressor()
    elif n in ('quantization', 'quant'):
        return GPUQuantizationCompressor()
    elif n in ('llmint8', 'llm_int8', 'llm.int8'):
        return GPULLMInt8Compressor()
    else:
        raise ValueError(
            f"Unknown GPU compressor '{name}'. "
            f"Choose from: topk, quantization, llmint8"
        )


# ============================================================================
# Self-test
# ============================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print(f"GPU compressor self-test  (device={device})")
    print("=" * 60)

    torch.manual_seed(42)
    tensor = torch.relu(torch.randn(2, 64, 8, 8)).to(device)
    original_bytes = tensor.numel() * 4
    print(f"Input: {tensor.shape}  {original_bytes} bytes  device={tensor.device}\n")

    # --- TopK ---
    for k in [0.5, 0.25, 0.125]:
        comp = GPUTopKCompressor()
        data = comp.compress(tensor, k)
        out  = comp.decompress(data, device=device)
        err  = (tensor - out).abs().mean().item()
        ratio = comp.get_compressed_size(data) / original_bytes
        assert out.shape == tensor.shape
        assert out.device.type == device.type
        print(f"[{comp.get_name()}] k={k:.3f}  ratio={ratio:.3f}  MAE={err:.6f}  ✓")

    print()

    # --- Quantization ---
    labels = {0.5: 'FP16', 0.25: 'INT8', 0.125: 'INT4', 0.0625: 'INT2'}
    for k in [0.5, 0.25, 0.125, 0.0625]:
        comp = GPUQuantizationCompressor()
        data = comp.compress(tensor, k)
        out  = comp.decompress(data, device=device)
        err  = (tensor - out).abs().mean().item()
        ratio = comp.get_compressed_size(data) / original_bytes
        assert out.shape == tensor.shape
        assert out.device.type == device.type
        print(f"[{comp.get_name()}] {labels[k]:<5}  k={k:.4f}  "
              f"ratio={ratio:.3f}  MAE={err:.6f}  ✓")

    print()

    # --- LLMInt8 ---
    configs = [
        [0.01, 'fp16', 'int8'],
        [0.05, 'fp16', 'int4'],
        [0.10, 'int8', 'int2'],
    ]
    for params in configs:
        comp = GPULLMInt8Compressor()
        data = comp.compress(tensor, params)
        out  = comp.decompress(data, device=device)
        err  = (tensor - out).abs().mean().item()
        ratio = data['compression_ratio']
        assert out.shape == tensor.shape
        assert out.device.type == device.type
        print(f"[{comp.get_name()}] k={params[0]:.2f} "
              f"out={params[1]} reg={params[2]}  "
              f"ratio={ratio:.3f}  MAE={err:.6f}  ✓")

    print("\nAll tests passed.")
