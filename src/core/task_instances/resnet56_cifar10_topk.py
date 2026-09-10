from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .. import resnet20
from ..accuracy_fn import AccuracyFunction, topk_sparsify
from ..task import InferenceTask
from ..toy_A import grad_oracle


_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_ROOT = _ROOT / "data"
_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default compute/communication profile (4-node / 3-link chain) when no trace path is set.
_DEFAULT_TAU = {
    0: 0.010589158860966563,
    1: 0.00960599142126739,
    2: 0.011260934406891465,
    3: 0.011232641758397222,
}
_DEFAULT_A = {
    0: 6553600.0,
    1: 3276800.0,
    2: 1638400.0,
}

_DEFAULT_TRUE_SAMPLES = 200
_DEFAULT_FAST_SAMPLES = 100
_DEFAULT_SIGMA = 0.05
_DEFAULT_N = 5
_DEFAULT_ETA_MIN = 0.02
_DEFAULT_ACCURACY_ESTIMATOR_MODE = "stein_estimator"
_DEFAULT_COMPRESSOR_NAME = "topk"
_DEFAULT_LLMINT8_POLICY = "fp16_int4"
_DEFAULT_LLMINT8_OUTLIER_PRECISION = "fp16"
_DEFAULT_LLMINT8_REGULAR_PRECISION = "int4"

from ..llmint8 import resolve_codec_execution_plan
from ..llmint8 import validate_llmint8_mapping_entries

try:
    from ..compressors import get_compressor  # type: ignore
except ModuleNotFoundError:
    get_compressor = None  # type: ignore[assignment]



def _env_path(name: str, default_path: Path | None = None) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    if default_path is not None:
        return default_path
    raise FileNotFoundError(
        "Environment variable {} must be set to a readable path.".format(name)
    )


def _resolve_accuracy_estimator_mode() -> str:
    value = os.environ.get("RESNET56_ACCURACY_ESTIMATOR_MODE", _DEFAULT_ACCURACY_ESTIMATOR_MODE)
    mode = str(value).strip().lower().replace(" ", "_")
    aliases = {
        "stein": "stein_estimator",
        "stein_estimator": "stein_estimator",
        "fitting": "fitting_model",
        "fitting_model": "fitting_model",
    }
    if mode not in aliases:
        raise ValueError(
            "Unsupported RESNET56_ACCURACY_ESTIMATOR_MODE={!r}. "
            "Expected one of: stein_estimator, fitting_model.".format(value)
        )
    return aliases[mode]


def _resolve_compressor_name(override_name: str | None = None) -> str:
    value = override_name if override_name is not None else os.environ.get("RESNET56_COMPRESSOR_NAME", _DEFAULT_COMPRESSOR_NAME)
    name = str(value).strip().lower()
    aliases = {
        "topk": "topk",
        "quant": "quantization",
        "quantization": "quantization",
        "llmint8": "llmint8",
        "llm_int8": "llmint8",
        "llm.int8": "llmint8",
    }
    if name not in aliases:
        raise ValueError(
            "Unsupported RESNET56_COMPRESSOR_NAME={!r}. Expected one of: topk, quantization, llmint8.".format(value)
        )
    return aliases[name]


def _resolve_fitting_model_path(compressor_name: str) -> Path:
    del compressor_name  # codec-specific defaults are not bundled; path must be set explicitly
    return _env_path("RESNET56_FITTING_MODEL_PATH")


def _resolve_llmint8_mapping_path() -> Path | None:
    explicit_path = os.environ.get("RESNET56_LLMINT8_MAPPING_PATH")
    if not explicit_path:
        return None
    path = Path(explicit_path).expanduser()
    return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()


def _resolve_llmint8_mapping_entries(num_links: int) -> list[Dict[str, Any]]:
    path = _resolve_llmint8_mapping_path()
    if path is None or not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("llmint8_eta_to_codec_mapping", {}).get("entries", payload.get("entries", []))
    if not entries and "policies" in payload:
        preferred_policy = str(os.environ.get("RESNET56_LLMINT8_POLICY", _DEFAULT_LLMINT8_POLICY)).strip().lower()
        for policy in payload.get("policies", []):
            if str(policy.get("llm_policy", "")).strip().lower() == preferred_policy:
                entries = policy.get("llmint8_eta_to_codec_mapping", {}).get("entries", [])
                if entries:
                    break
        if not entries:
            for policy in payload.get("policies", []):
                entries = policy.get("llmint8_eta_to_codec_mapping", {}).get("entries", [])
                if entries:
                    break
    return validate_llmint8_mapping_entries(entries, num_links=num_links)


class _Poly3FittingAccuracyAdapter:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path).resolve()
        with self.model_path.open("rb") as handle:
            state = pickle.load(handle)

        self.model = state["model"]
        self.scaler = state["scaler"]
        self.poly = state.get("poly")
        self.model_type = str(state["model_type"]).strip().lower()
        self.n_features = int(state["n_features"])
        self.feature_names = state.get("feature_names")

        if self.model_type != "poly3":
            raise ValueError(
                "Fitting model {!s} has model_type={!r}; Phase 1 currently supports only 'poly3'.".format(
                    self.model_path,
                    state["model_type"],
                )
            )
        if self.poly is None:
            raise ValueError("Fitting model {!s} is missing polynomial features metadata.".format(self.model_path))

    def _predict_raw(self, eta: np.ndarray) -> float:
        x = np.asarray(eta, dtype=float).reshape(1, -1)
        if x.shape[1] != self.n_features:
            raise ValueError("Expected {} eta values, got {}".format(self.n_features, x.shape[1]))
        z = self.scaler.transform(x)
        z = self.poly.transform(z)
        return float(self.model.predict(z)[0])

    def predict(self, eta: np.ndarray) -> float:
        return float(np.clip(self._predict_raw(np.asarray(eta, dtype=float).reshape(-1)), 0.0, 1.0))

    def gradient(self, eta: np.ndarray) -> np.ndarray:
        x = np.asarray(eta, dtype=float).reshape(-1)
        if x.shape[0] != self.n_features:
            raise ValueError("Expected {} eta values, got {}".format(self.n_features, x.shape[0]))

        mean = np.asarray(getattr(self.scaler, "mean_", np.zeros(self.n_features)), dtype=float)
        scale = np.asarray(getattr(self.scaler, "scale_", np.ones(self.n_features)), dtype=float)
        scale = np.where(scale == 0.0, 1.0, scale)
        z = (x - mean) / scale

        coef = np.asarray(self.model.coef_, dtype=float).reshape(-1)
        powers = np.asarray(self.poly.powers_, dtype=int)
        grad_z = np.zeros(self.n_features, dtype=float)
        for feature_idx in range(self.n_features):
            partial = 0.0
            for term_idx, power_vec in enumerate(powers):
                exponent = int(power_vec[feature_idx])
                if exponent == 0:
                    continue
                term = coef[term_idx] * exponent
                for dim_idx, dim_power in enumerate(power_vec):
                    p = int(dim_power)
                    if dim_idx == feature_idx:
                        if p - 1 > 0:
                            term *= z[dim_idx] ** (p - 1)
                    elif p > 0:
                        term *= z[dim_idx] ** p
                partial += term
            grad_z[feature_idx] = partial

        grad = grad_z / scale
        raw = self._predict_raw(x)
        if raw <= 0.0 or raw >= 1.0:
            return np.zeros_like(grad)
        return grad.astype(float)


class Phase1CompressedModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        eta: torch.Tensor,
        *,
        cut_modules=None,
        compressor_name: str = "topk",
        llmint8_mapping_entries=None,
        llmint8_outlier_precision: str = "fp16",
        llmint8_regular_precision: str = "int4",
    ):
        super().__init__()
        self.base_model = base_model
        self.eta = eta.clone()
        self.compressor_name = str(compressor_name).strip().lower()
        self.llmint8_mapping_entries = list(llmint8_mapping_entries or [])
        self.llmint8_outlier_precision = str(llmint8_outlier_precision)
        self.llmint8_regular_precision = str(llmint8_regular_precision)
        if self.compressor_name not in ("topk", "identity", "none") and get_compressor is None:
            raise ModuleNotFoundError(
                "compressors module is required for compressor '{}' but was not found. "
                "Use topk mode or install/provide the compressors module.".format(self.compressor_name)
            )
        self._compressor = (
            None
            if self.compressor_name in ("topk", "identity", "none")
            else get_compressor(self.compressor_name)
        )
        self._compression_params_list = [float(item) for item in eta.detach().cpu().tolist()]

        if cut_modules is None:
            cut_modules = [m for m in base_model.modules() if isinstance(m, (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid))]
        self.cut_modules = list(cut_modules)
        self.d = len(self.cut_modules)
        assert len(eta) == self.d, f"eta length {len(eta)} does not match {self.d} cut-points"

        for i, m in enumerate(self.cut_modules):
            m.register_forward_hook(self._make_hook(i))

    def _make_hook(self, i: int):
        def hook(module, input, output):
            compression_param = self._compression_params_list[i]
            if self.compressor_name == "topk":
                return topk_sparsify(output, self.eta[i].item())
            if self.compressor_name in ("identity", None) or compression_param is None:
                return output
            tensor_cpu = output.detach().cpu()
            compressed = self._compressor.compress(tensor_cpu, compression_param)
            restored = self._compressor.decompress(compressed)
            return restored.to(device=output.device, dtype=output.dtype)

        return hook

    def set_eta(self, eta: torch.Tensor):
        assert len(eta) == self.d
        self.eta = eta.clone()
        eta_list = [float(item) for item in eta.detach().cpu().tolist()]
        if self.compressor_name == "llmint8":
            execution_plan = resolve_codec_execution_plan(
                codec_name="llmint8",
                eta=eta_list,
                outlier_precision=self.llmint8_outlier_precision,
                regular_precision=self.llmint8_regular_precision,
                llmint8_mapping_entries=self.llmint8_mapping_entries,
            )
            self._compression_params_list = list(execution_plan["compression_params_list"])
        else:
            self._compression_params_list = eta_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


class Phase1InferenceTaskCallables:
    def __init__(
        self,
        cm: Phase1CompressedModel,
        test_loader: DataLoader,
        sigma: float = 0.05,
        N: int = 50,
        n_samples: int = 512,
        n_samples_true: int | None = None,
        flat_theta: torch.Tensor = None,
        eta_min=None,
        device=None,
    ):
        self.cm = cm
        self.sigma = sigma
        self.N = N
        self.flat_theta = flat_theta
        self.device = device or next(cm.parameters()).device
        if eta_min is None:
            eta_min = torch.zeros(cm.d, dtype=torch.float32)
        self.eta_min = torch.as_tensor(eta_min, dtype=torch.float32).reshape(-1)
        if int(self.eta_min.numel()) != int(cm.d):
            raise ValueError("Expected {} eta_min values, got {}".format(cm.d, self.eta_min.numel()))
        self.A = AccuracyFunction(cm, test_loader, n_samples=n_samples, device=self.device)
        self.A_true = AccuracyFunction(cm, test_loader, n_samples=n_samples_true, device=self.device)

    def _clamp_eta_torch(self, eta: torch.Tensor) -> torch.Tensor:
        eta_t = eta.to(dtype=torch.float32).reshape(-1)
        eta_min = self.eta_min.to(device=eta_t.device)
        eta_t = torch.maximum(eta_t, eta_min)
        return torch.minimum(eta_t, torch.ones_like(eta_t))

    def _acc_torch(self, eta: torch.Tensor) -> torch.Tensor:
        return self.A(self._clamp_eta_torch(eta), flat_theta=self.flat_theta)

    def _grad_torch(self, eta: torch.Tensor) -> torch.Tensor:
        return grad_oracle(self._acc_torch, self._clamp_eta_torch(eta), sigma=self.sigma, N=self.N)

    def _acc_true_torch(self, eta: torch.Tensor) -> torch.Tensor:
        return self.A_true(self._clamp_eta_torch(eta), flat_theta=self.flat_theta)

    def accuracy_callable(self, eta_vec: np.ndarray) -> float:
        eta = torch.tensor(eta_vec, dtype=torch.float32)
        return self._acc_torch(eta).item()

    def gradient_callable(self, eta_vec: np.ndarray) -> np.ndarray:
        eta = torch.tensor(eta_vec, dtype=torch.float32)
        return self._grad_torch(eta).numpy()

    def accuracy_callable_true(self, eta_vec: np.ndarray) -> float:
        eta = torch.tensor(eta_vec, dtype=torch.float32)
        return self._acc_true_torch(eta).item()


def _load_profile(trace_path: Path | None) -> Tuple[Dict[int, float], Dict[int, float]]:
    if trace_path is None:
        return dict(_DEFAULT_TAU), dict(_DEFAULT_A)

    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace = payload["trace"]
    activation_bytes = np.asarray(trace["activation_bytes"], dtype=float)
    tau_compute = np.asarray(trace["tau_compute_node_sec"], dtype=float)

    a_med = np.median(activation_bytes, axis=0)
    tau_med = np.median(tau_compute, axis=0)

    a = {int(idx): float(a_med[idx]) for idx in range(a_med.shape[0])}
    tau = {int(idx): float(tau_med[idx]) for idx in range(tau_med.shape[0])}
    return tau, a


def _load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = resnet20.resnet56()
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    normalized = {}
    for key, value in state_dict.items():
        normalized[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(normalized)
    model.to(device)
    model.eval()
    return model


def _build_test_loader(data_root: Path, batch_size: int = 100, download: bool = False) -> DataLoader:
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    dataset = datasets.CIFAR10(root=str(data_root), train=False, download=download, transform=tf)
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=False)


def setup_model_and_callables(*, compressor_name: str | None = None):
    device = torch.device(os.environ.get("RESNET56_PHASE1_DEVICE", _DEFAULT_DEVICE))
    checkpoint_path = _env_path("RESNET56_PHASE1_CHECKPOINT")
    data_root = _env_path("RESNET56_PHASE1_DATA_ROOT", _DEFAULT_DATA_ROOT)
    fast_samples = int(os.environ.get("RESNET56_PHASE1_FAST_SAMPLES", str(_DEFAULT_FAST_SAMPLES)))
    true_samples = int(os.environ.get("RESNET56_PHASE1_TRUE_SAMPLES", str(_DEFAULT_TRUE_SAMPLES)))
    sigma = float(os.environ.get("RESNET56_PHASE1_SIGMA", str(_DEFAULT_SIGMA)))
    stein_n = int(os.environ.get("RESNET56_PHASE1_STEIN_N", str(_DEFAULT_N)))
    download = os.environ.get("RESNET56_PHASE1_DOWNLOAD", "0").strip() in {"1", "true", "True"}
    compressor_name = _resolve_compressor_name(compressor_name)

    model = _load_model(checkpoint_path, device)
    test_loader = _build_test_loader(data_root=data_root, batch_size=100, download=download)
    llmint8_mapping_entries = _resolve_llmint8_mapping_entries(num_links=3) if compressor_name == "llmint8" else []
    llmint8_outlier_precision = str(
        os.environ.get("RESNET56_LLMINT8_OUTLIER_PRECISION", _DEFAULT_LLMINT8_OUTLIER_PRECISION)
    )
    llmint8_regular_precision = str(
        os.environ.get("RESNET56_LLMINT8_REGULAR_PRECISION", _DEFAULT_LLMINT8_REGULAR_PRECISION)
    )

    # Three transport links in a 4-node chain: after layer1, layer2, layer3.
    cut_modules = [model.layer1, model.layer2, model.layer3]
    eta_init = torch.ones(len(cut_modules))
    eta_floor = float(os.environ.get("RESNET56_PHASE1_ETA_MIN", str(_DEFAULT_ETA_MIN)))
    cm = Phase1CompressedModel(
        model,
        eta_init,
        cut_modules=cut_modules,
        compressor_name=compressor_name,
        llmint8_mapping_entries=llmint8_mapping_entries,
        llmint8_outlier_precision=llmint8_outlier_precision,
        llmint8_regular_precision=llmint8_regular_precision,
    )
    api = Phase1InferenceTaskCallables(
        cm,
        test_loader,
        sigma=sigma,
        N=stein_n,
        n_samples=fast_samples,
        n_samples_true=true_samples,
        flat_theta=None,
        eta_min=[eta_floor] * len(cut_modules),
        device=device,
    )

    extra = {
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "data_root": str(data_root),
        "fast_samples": int(fast_samples),
        "true_samples": int(true_samples),
        "compressor_name": compressor_name,

    }
    return api, extra


def make_task(
    task_id: int,
    api: InferenceTaskCallables,
    w_k: float = 1.0,
    R_k: float = 10.0,
    compressor_name: str | None = None,
) -> InferenceTask:
    trace_env = os.environ.get("RESNET56_PHASE1_TRACE_PATH")
    trace_path = Path(trace_env).expanduser().resolve() if trace_env else None
    tau, a = _load_profile(trace_path)
    eta_floor = float(os.environ.get("RESNET56_PHASE1_ETA_MIN", str(_DEFAULT_ETA_MIN)))
    accuracy_mode = _resolve_accuracy_estimator_mode()
    compressor_name = _resolve_compressor_name(compressor_name)

    fitting_model_path: Path | None = None
    if accuracy_mode == "fitting_model":
        fitting_model_path = _resolve_fitting_model_path(compressor_name)
        if not fitting_model_path.exists():
            raise FileNotFoundError(
                "Requested fitting model was not found: {!s}. "
                "Set RESNET56_FITTING_MODEL_PATH.".format(
                    fitting_model_path
                )
            )
        acc_adapter = _Poly3FittingAccuracyAdapter(fitting_model_path)
        accuracy_callable = acc_adapter.predict
        gradient_callable = acc_adapter.gradient
    else:
        accuracy_callable = api.accuracy_callable
        gradient_callable = api.gradient_callable

    task = InferenceTask(
        task_id=task_id,
        b_k=0,
        L_k=4,
        tau=tau,
        a=a,
        eta_min={0: eta_floor, 1: eta_floor, 2: eta_floor},
        R_k_callable=lambda t, _R=R_k: float(_R),
        w_k=w_k,
        accuracy_callable=accuracy_callable,
        accuracy_callable_true=api.accuracy_callable_true,
        gradient_callable=gradient_callable,
    )
    task.accuracy_estimator_mode = accuracy_mode
    task.fitting_model_type = "poly3" if fitting_model_path is not None else None
    task.fitting_model_path = str(fitting_model_path) if fitting_model_path is not None else None
    task.compressor_name = compressor_name
    return task
