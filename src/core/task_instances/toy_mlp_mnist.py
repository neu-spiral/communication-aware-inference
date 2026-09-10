"""
Toy MLP (3 epochs trained) on MNIST with Stein gradient oracle.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ..accuracy_fn import topk_sparsify
from ..compressors import get_compressor
from ..llmint8 import resolve_codec_execution_plan
from ..stein_oracles import MLP, get_flat_params
from ..task import InferenceTask
from ..task_callables import InferenceTaskCallables

# Bytes per compressed activation tensor on each link (float32), for a batch of B
# images. Cuts follow ReLU outputs: (B, 128) then (B, 64). Match ResNet trace
# convention (B = 100).
_MLP_COMM_BATCH = 100
_MLP_A_BYTES = {
    0: float(_MLP_COMM_BATCH * 128 * 4),
    1: float(_MLP_COMM_BATCH * 64 * 4),
}


class _MLPCompressedModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        eta: torch.Tensor,
        *,
        compressor_name: str = "topk",
        cut_modules=None,
        llmint8_outlier_precision: str = "fp16",
        llmint8_regular_precision: str = "int4",
    ):
        super().__init__()
        self.base_model = base_model
        self.eta = eta.clone()
        self.compressor_name = str(compressor_name).strip().lower()
        self.llmint8_outlier_precision = str(llmint8_outlier_precision)
        self.llmint8_regular_precision = str(llmint8_regular_precision)
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
        def hook(_module, _input, output):
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
                llmint8_mapping_entries=[],
            )
            self._compression_params_list = list(execution_plan["compression_params_list"])
        else:
            self._compression_params_list = eta_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


def setup_model_and_callables(*, compressor_name: str = "topk"):
    """
    Trains a small MLP on MNIST and returns (api, flat_theta) for building tasks.
    api.accuracy_callable(eta_np) -> float, api.gradient_callable(eta_np) -> np.ndarray
    """
    torch.manual_seed(42)
    device = torch.device("cpu")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=tf),
        batch_size=128,
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=tf),
        batch_size=256,
        shuffle=False,
    )

    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Training MLP (3 epochs)...")
    for _ in range(3):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()

    flat_theta = get_flat_params(model)

    eta_init = torch.ones(2)
    cm = _MLPCompressedModel(model, eta_init, compressor_name=compressor_name)
    api = InferenceTaskCallables(
        cm,
        test_loader,
        sigma=0.05,
        N=50,
        n_samples=512,
        flat_theta=None,
        device=device,
    )

    return api, flat_theta


def make_task(
    task_id: int,
    api: InferenceTaskCallables,
    w_k: float = 1.0,
    R_k: float = 10.0,
    compressor_name: str = "topk",
) -> InferenceTask:
    task = InferenceTask(
        task_id=task_id,
        b_k=0,
        L_k=3,
        tau={0: 0.01, 1: 0.01, 2: 0.01},
        a=dict(_MLP_A_BYTES), #10 , 20
        eta_min={0: 0.01, 1: 0.01},
        R_k_callable=lambda t, _R=R_k: float(_R),
        w_k=w_k,
        accuracy_callable=api.accuracy_callable,
        accuracy_callable_true=api.accuracy_callable_true,
        gradient_callable=api.gradient_callable,
    )
    task.compressor_name = str(compressor_name).strip().lower()
    return task
