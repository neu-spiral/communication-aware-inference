"""
resnet_task_callables.py
========================
ResNet56 / CIFAR-10 accuracy backend for InferenceTask.

Provides the same three-callable interface as InferenceTaskCallables
(MLP/MNIST) and LLMInferenceTaskCallables (LLM/MMLU):

    api.accuracy_callable(eta_np)       -> float   (fast subset)
    api.gradient_callable(eta_np)       -> np.ndarray
    api.accuracy_callable_true(eta_np)  -> float   (full test set)

How eta maps to compression
---------------------------
eta is a vector of length L = len(CUTPOINTS), one value per transfer link.

    eta[i] = 1.0  -> no compression on link i  (passthrough)
    eta[i] < 1.0  -> compress with ratio eta[i] using the chosen compressor

For TopKCompressor   : eta[i] = keep-fraction (k_ratio)
For QuantizationCompressor: eta[i] snapped to nearest supported level
                           (0.5->FP16, 0.25->INT8, 0.125->INT4, 0.0625->INT2)
For LLMInt8Compressor: eta[i] = outlier fraction, with 'fp16'/'int8' defaults

Payload sizes for the delay model
----------------------------------
The actual compressed-activation size in bytes depends on the chosen
compressor and eta.  ResNetTaskCallables.measured_a() returns a dict
{ link_index -> bytes } by running one calibration forward pass, so that
InferenceTask's delay constraint uses real numbers rather than toy values.

Topology produced by build_resnet_backend_from_env
---------------------------------------------------
CUTPOINTS = [8, 14, 21]  (default, matches resnet_local_experiment.py)
produces 4 partitions and 3 transfer links:

    Partition 0  [stem + layer1.0..layer1.6]   Node 0
    Partition 1  [layer1.7 + layer1.8 + layer2.0..layer2.3]   Node 1
    Partition 2  [layer2.4..layer2.8 + layer3.0 + layer3.1]   Node 2
    Partition 3  [layer3.2..layer3.8 + head]   Node 3

    Links: 0 (Node0->Node1), 1 (Node1->Node2), 2 (Node2->Node3)
    M = 4  (nodes), L = 3 (links = len(CUTPOINTS))

Environment variables
---------------------
    RESNET_CHECKPOINT   path to resnet56-4bfd9763.th  (required)
    RESNET_DATA_ROOT    CIFAR-10 root directory        (default: ./data)
    RESNET_CUTPOINTS    comma-separated int list       (default: 8,14,21)
    RESNET_BATCH_SIZE   int                            (default: 100)
    RESNET_FAST_BATCHES int, batches for fast eval     (default: 2)
    RESNET_DOWNLOAD     1/0                            (default: 1)
    COMPRESSOR_TYPE     topk | quantization | llmint8  (default: topk)
    RESNET_GRAD_SIGMA   Stein smoothing scale          (default: 0.05)
    RESNET_GRAD_N       Stein sample pairs             (default: 20)

Quick start
-----------
    export RESNET_CHECKPOINT=./resnet56-4bfd9763.th
    export INFERENCE_OPTIMIZER_BACKEND=resnet
    python main.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet20
from .compressor_registry import (
    get_compressor      as _registry_get_compressor,
    is_gpu_compressor   as _is_gpu_compressor,
    is_gpu_backend      as _is_gpu_backend,
    describe_compressor as _describe_compressor,
    COMPRESSOR_BACKEND  as _COMPRESSOR_BACKEND,
)
from .compressors import BaseCompressor, get_compressor
from .toy_A import grad_oracle


# ============================================================================
# Model loading helpers
# ============================================================================

class _ResNetHead(nn.Module):
    """avgpool + flatten + linear grouped as a single unit."""
    def __init__(self, avgpool, flatten, linear):
        super().__init__()
        self.avgpool = avgpool
        self.flatten = flatten
        self.linear  = linear

    def forward(self, x):
        return self.linear(self.flatten(self.avgpool(x)))


class _ResNetUnitPartition(nn.Module):
    """Ordered sequence of ResNet units (stem / BasicBlocks / head)."""
    def __init__(self, units: List[nn.Module]):
        super().__init__()
        self.units = nn.ModuleList(units)

    def forward(self, x):
        for m in self.units:
            x = m(x)
        return x


def _normalize_state_dict(sd):
    from collections import OrderedDict
    out = OrderedDict()
    for k, v in sd.items():
        out[k[7:] if k.startswith('module.') else k] = v
    return out


def _load_resnet56(checkpoint_path: str, device: torch.device) -> nn.Module:
    model = resnet20.resnet56()
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd    = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(_normalize_state_dict(sd))
    model.to(device).eval()
    return model


def _build_units(model: nn.Module):
    """Decompose ResNet56 into 29 addressable units (same order as resnet_local_experiment)."""
    units, names = [], []
    units.append(nn.Sequential(model.conv1, model.bn1, model.relu))
    names.append('stem')
    for i, blk in enumerate(model.layer1):
        units.append(blk);  names.append(f'layer1.{i}')
    for i, blk in enumerate(model.layer2):
        units.append(blk);  names.append(f'layer2.{i}')
    for i, blk in enumerate(model.layer3):
        units.append(blk);  names.append(f'layer3.{i}')
    units.append(_ResNetHead(model.avgpool, model.flatten, model.linear))
    names.append('head')
    return units, names


def _build_partitions(units, cutpoints: List[int], device: torch.device):
    """Split unit list at cutpoints -> list of _ResNetUnitPartition on device."""
    n = len(units)
    validated = sorted(int(c) for c in cutpoints)
    if len(set(validated)) != len(validated):
        raise ValueError(f"Duplicate cutpoints: {validated}")
    if validated and (validated[0] <= 0 or validated[-1] >= n):
        raise ValueError(f"Cutpoints out of range (1..{n-1}): {validated}")
    boundaries = [0] + validated + [n]
    parts = []
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        p = _ResNetUnitPartition(units[s:e]).to(device)
        p.eval()
        parts.append(p)
    return parts


# ============================================================================
# CIFAR-10 loading
# ============================================================================

def _load_cifar10_batches(
    data_root: str,
    batch_size: int,
    max_batches: Optional[int],
    download: bool,
) -> List[Dict]:
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = torchvision.datasets.CIFAR10(
        root=str(data_root), train=False, download=download, transform=tf
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    batches = []
    for imgs, lbls in loader:
        batches.append({'images': imgs, 'labels': lbls})
        if max_batches is not None and len(batches) >= max_batches:
            break
    return batches


# ============================================================================
# Core partitioned inference  (mirrors simulate_partitioned_batch)
# ============================================================================

def _run_partitioned(
    partitions: List[_ResNetUnitPartition],
    images: torch.Tensor,
    compressor,
    eta_vec: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Run a single batch through all partitions with compress/decompress between them.

    Returns (logits, transfer_stats_list).
    transfer_stats[i] has keys: original_bytes, compressed_bytes, payload_ratio.
    """
    x = images.to(device)
    transfer_stats = []

    with torch.no_grad():
        for idx, partition in enumerate(partitions):
            x = partition(x)

            # Compress the activation between this partition and the next
            if idx < len(partitions) - 1:
                eta = float(eta_vec[idx])
                # Clamp eta to [0, 1] — the Stein oracle perturbs eta with
                # x ± sigma*z which can push it outside this range.
                # LLMInt8 uses eta as an outlier_ratio passed to quantile()
                # which requires its argument in [0, 1]; TopK/Quant also
                # behave correctly only within this range.
                eta = max(0.0, min(1.0, eta))
                original_bytes = int(x.detach().numel() * x.element_size())

                if eta >= 1.0:
                    # No compression — pass through
                    compressed_bytes = original_bytes
                else:
                    # Build compressor params: TopK/Quant take float, LLMInt8 takes list
                    name = compressor.get_name().lower()
                    if 'llmint8' in name or 'llm' in name:
                        params = [eta, 'fp16', 'int8']
                    else:
                        params = eta
                    

                    if _is_gpu_compressor(compressor):
                        # GPU-native path — tensor stays on device throughout
                        compressed = compressor.compress(x.detach(), params)
                        compressed_bytes = int(
                            compressor.get_compressed_size(compressed)
                            if hasattr(compressor, 'get_compressed_size')
                            else original_bytes
                        )
                        x = compressor.decompress(compressed, device=device)
                    else:
                        # CPU path — original behaviour
                        tensor_cpu = x.detach().cpu()
                        compressed = compressor.compress(tensor_cpu, params)
                        compressed_bytes = int(
                            compressor.get_compressed_size(compressed)
                            if hasattr(compressor, 'get_compressed_size')
                            else compressed.get('compressed_bytes', original_bytes)
                        )
                        x = compressor.decompress(compressed, device='cpu').to(device)

                payload_ratio = min(1.0, compressed_bytes / original_bytes) if original_bytes > 0 else 1.0
                transfer_stats.append({
                    'original_bytes':   original_bytes,
                    'compressed_bytes': compressed_bytes,
                    'payload_ratio':    payload_ratio,
                })

    return x, transfer_stats


# ============================================================================
# ResNetTaskCallables  — the public API class
# ============================================================================

class ResNetTaskCallables:
    """
    Accuracy / gradient callables for InferenceTask backed by
    ResNet56 on CIFAR-10 with configurable activation compression.

    Parameters
    ----------
    checkpoint_path : str
        Path to resnet56-4bfd9763.th (or any compatible checkpoint).
    data_root : str
        Root directory for CIFAR-10 (downloaded if absent and download=True).
    cutpoints : list of int
        Unit indices at which to insert transfer links.
        Default [8, 14, 21] gives 3 links (4 partitions).
    batch_size : int
        Batch size for CIFAR-10 evaluation.
    fast_batches : int or None
        Number of batches used by accuracy_callable (fast, noisy).
        None = full test set (same as accuracy_callable_true).
    download : bool
        Whether to auto-download CIFAR-10 if not present.
    compressor : str or BaseCompressor
        Compression backend: 'topk' | 'quantization' | 'llmint8', or
        a compressors.BaseCompressor instance.
    sigma : float
        Stein smoothing scale for the gradient oracle.
    N : int
        Number of antithetic Stein sample pairs for the gradient oracle.
    device : torch.device or None
        Compute device. Defaults to CUDA if available, else CPU.
    """

    def __init__(
        self,
        checkpoint_path: str,
        data_root: str = './data',
        cutpoints: Optional[List[int]] = None,
        batch_size: int = 100,
        fast_batches: Optional[int] = 2,
        download: bool = True,
        compressor = 'topk',
        sigma: float = 0.05,
        N: int = 20,
        device: Optional[torch.device] = None,
    ):
        self.cutpoints   = cutpoints if cutpoints is not None else [8, 14, 21]
        self.sigma       = sigma
        self.N           = N
        self.device      = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Resolve compressor via registry — respects COMPRESSOR_BACKEND
        if isinstance(compressor, str):
            self.compressor = _registry_get_compressor(compressor)
        else:
            self.compressor = compressor

        # ── Clear printout ─────────────────────────────────────────────────
        print(f"[ResNet backend] device={self.device}", flush=True)
        print(_describe_compressor(self.compressor, 'ResNet backend'), flush=True)
        print(f"[ResNet backend] cutpoints={self.cutpoints}  "
              f"({len(self.cutpoints)} links, {len(self.cutpoints)+1} partitions)",
              flush=True)

        # Load model and build partitions
        self._model = _load_resnet56(checkpoint_path, self.device)
        units, _ = _build_units(self._model)
        self._partitions = _build_partitions(units, self.cutpoints, self.device)

        # Load CIFAR-10 batches upfront (they fit in RAM)
        print(f"[ResNet backend] Loading CIFAR-10 from {data_root} ...")
        all_batches = _load_cifar10_batches(data_root, batch_size, None, download)
        self._fast_batches = (
            all_batches[:fast_batches] if fast_batches is not None else all_batches
        )
        self._full_batches = all_batches
        print(f"[ResNet backend] {len(all_batches)} total batches  "
              f"({len(self._fast_batches)} used for fast eval)")

        # Number of links = number of cutpoints
        self.num_links = len(self.cutpoints)

    # ------------------------------------------------------------------ #
    #  Measured payload sizes  (used to populate InferenceTask.a)         #
    # ------------------------------------------------------------------ #

    def measured_a(self, eta: Optional[np.ndarray] = None) -> Dict[int, float]:
        """
        Return a dict {link_index: compressed_bytes_per_sample} by running
        one calibration batch at the given eta values.

        If eta is None, uses eta=1.0 (uncompressed) to get raw activation sizes.
        These values can be passed directly as InferenceTask.a so that the
        delay constraint uses real ResNet activation sizes rather than toy numbers.

        Example
        -------
            a = api.measured_a()          # raw sizes
            a = api.measured_a(eta_vec)   # compressed sizes at operating point
        """
        if eta is None:
            eta = np.ones(self.num_links)
        batch = self._fast_batches[0]
        _, stats = _run_partitioned(
            self._partitions, batch['images'], self.compressor, eta, self.device
        )
        n = int(batch['images'].shape[0])
        return {i: stats[i]['compressed_bytes'] / n for i in range(len(stats))}

    # ------------------------------------------------------------------ #
    #  Internal evaluation over a batch list                               #
    # ------------------------------------------------------------------ #

    def _evaluate(self, eta_vec: np.ndarray, batches: List[Dict]) -> float:
        """Top-1 accuracy over *batches* at compression ratios *eta_vec*."""
        correct = total = 0
        for batch in batches:
            logits, _ = _run_partitioned(
                self._partitions, batch['images'], self.compressor, eta_vec, self.device
            )
            preds    = logits.argmax(dim=1).cpu()
            labels   = batch['labels']
            correct += int((preds == labels).sum().item())
            total   += int(labels.numel())
        return float(correct) / float(total) if total > 0 else 0.0

    # ------------------------------------------------------------------ #
    #  Public callables — match InferenceTaskCallables interface exactly   #
    # ------------------------------------------------------------------ #

    def accuracy_callable(self, eta_vec: np.ndarray) -> float:
        """
        Fast accuracy estimate over a small subset of CIFAR-10 batches.
        Called repeatedly by the Stein gradient oracle (O(2N) times per slot).

        np.ndarray (shape [L,]) -> float in [0, 1]
        """
        return self._evaluate(eta_vec, self._fast_batches)

    def gradient_callable(self, eta_vec: np.ndarray) -> np.ndarray:
        """
        Stein zeroth-order gradient of accuracy w.r.t. eta.

        np.ndarray (shape [L,]) -> np.ndarray (shape [L,])
        """
        eta_t = torch.tensor(eta_vec, dtype=torch.float32)

        def f(e: torch.Tensor) -> torch.Tensor:
            acc = self._evaluate(e.detach().cpu().numpy(), self._fast_batches)
            return torch.tensor(acc, dtype=torch.float32)

        g = grad_oracle(f, eta_t, sigma=self.sigma, N=self.N)
        return g.detach().cpu().numpy()

    def accuracy_callable_true(self, eta_vec: np.ndarray) -> float:
        """
        Full-dataset accuracy — used by the simulation runner for logging.

        np.ndarray (shape [L,]) -> float in [0, 1]
        """
        return self._evaluate(eta_vec, self._full_batches)


# ============================================================================
# Environment-driven factory
# ============================================================================

def build_resnet_backend_from_env() -> ResNetTaskCallables:
    """
    Construct a ResNetTaskCallables from environment variables.

    Required
    --------
    RESNET_CHECKPOINT   path to the ResNet56 .th checkpoint file

    Optional
    --------
    RESNET_DATA_ROOT    CIFAR-10 root directory     (default: ./data)
    RESNET_CUTPOINTS    e.g. "8,14,21"              (default: 8,14,21)
    RESNET_BATCH_SIZE   int                          (default: 100)
    RESNET_FAST_BATCHES int or 'all'                 (default: 2)
    RESNET_DOWNLOAD     1 / 0                        (default: 1)
    COMPRESSOR_TYPE     topk | quantization | llmint8   (default: topk)
    RESNET_GRAD_SIGMA   float                        (default: 0.05)
    RESNET_GRAD_N       int                          (default: 20)
    """
    checkpoint = os.environ.get('RESNET_CHECKPOINT', '').strip()
    if not checkpoint:
        raise ValueError(
            "RESNET_CHECKPOINT environment variable must point to the "
            "ResNet56 checkpoint file (e.g. resnet56-4bfd9763.th)."
        )
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    data_root    = os.environ.get('RESNET_DATA_ROOT', './data').strip()
    compressor   = os.environ.get('COMPRESSOR_TYPE', 'topk').strip().lower()
    batch_size   = int(os.environ.get('RESNET_BATCH_SIZE', '100'))
    download     = os.environ.get('RESNET_DOWNLOAD', '1').strip() in ('1', 'true', 'yes')
    sigma        = float(os.environ.get('RESNET_GRAD_SIGMA', '0.05'))
    N            = int(os.environ.get('RESNET_GRAD_N', '20'))

    raw_cp = os.environ.get('RESNET_CUTPOINTS', '8,14,21').strip()
    cutpoints = [int(x.strip()) for x in raw_cp.split(',') if x.strip()]

    raw_fb = os.environ.get('RESNET_FAST_BATCHES', '2').strip().lower()
    fast_batches = None if raw_fb == 'all' else int(raw_fb)

    return ResNetTaskCallables(
        checkpoint_path = checkpoint,
        data_root       = data_root,
        cutpoints       = cutpoints,
        batch_size      = batch_size,
        fast_batches    = fast_batches,
        download        = download,
        compressor      = compressor,
        sigma           = sigma,
        N               = N,
    )
