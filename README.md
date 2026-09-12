# Inference Optimizer

Offline simulation code accompanying:

**Communication-Aware Model Distributed Inference via Latent Representation Compression**  
*(MobiHoc 2026)*

> We study optimization of distributed model inference over resource-constrained edge resources. We propose a framework that optimizes the trade-off between model accuracy and communication costs by controlling latent representation compression to meet strict Quality of Service (QoS) throughput targets. For settings with known channel state information (CSI), we derive a closed-form optimal solution for single tasks and reduce the multi-task problem to a convex optimization program characterized by a per-link water-filling strategy. We extend these to handle unpredictable environments via a stochastic dual descent algorithm that relies only on causal channel estimates. We provide Lyapunov-based proofs demonstrating that our approach strictly satisfies long-term delay constraints while achieving a bounded optimality gap. Our results offer a robust, scalable blueprint for maximizing the performance of pipelined AI tasks in dynamic, resource-constrained distributed systems. We verify the effectiveness of our proposed framework through simulations and experiments with real edge devices.

Given edge nodes, pipelined inference tasks, and time-varying link capacities, the code chooses per-link compression ratios `eta` to maximize accuracy subject to long-term throughput / delay QoS. It includes CSI-aware optima and CSI-oblivious stochastic dual descent, plus baselines.

## Companion repositories

This repository is the **offline** half of the paper: the optimizer library and
the simulation harness. The online experiments run on real edge hardware, and
each testbed has its own repository:

| Testbed | Repository | Scope |
|---|---|---|
| Jetson | [`UIC-Networking-Research-Lab/CAMDI_RC_MobiHoc2026_Fitting_and_Jetson`](https://github.com/UIC-Networking-Research-Lab/CAMDI_RC_MobiHoc2026_Fitting_and_Jetson) | Four-node Jetson deployment, plus the accuracy-function fitting and trace-collection code behind `assets/`. |
| Raspberry Pi | [`neu-spiral/rasp_compression`](https://github.com/neu-spiral/rasp_compression) | Pipelined Gemma-2 2B inference with activation compression across a Raspberry Pi cluster over WiFi. |
| PRESCIENT (Ohio State) | [`PRESCIENT-osu/DNN-comm-compression`](https://github.com/PRESCIENT-osu/DNN-comm-compression) | Pipelined distributed inference over a programmable wide-area network, with configurable per-link bandwidth, delay, and loss. |

The Jetson repository is where the fitted accuracy models and measured traces
bundled in [`assets/`](assets/README.md) come from, and it documents how to
regenerate them. For an interactive view of the trade-off this paper optimizes,
[`clarayliu09/jarvis-visualization-demo`](https://github.com/clarayliu09/jarvis-visualization-demo)
replays Raspberry Pi traces across compression ratios from `eta = 0.1` to `1.0`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run scripts from the repository root so `from src...` imports resolve.

`requirements.txt` is unpinned, so `pip` installs the current PyTorch release,
whose default wheel targets the newest CUDA. If your driver is older,
`torch.cuda.is_available()` returns `False` and every GPU task silently runs on
CPU; check `nvidia-smi` and install a matching wheel first, e.g.
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`.

## Public entry points

| Script | Role |
|--------|------|
| `compare_to_baselines.py` | Single-task offline sims vs baselines → NPZ + plots |
| `compare_to_baselines_multi.py` | Multi-task offline sims vs baselines → NPZ + plots |
| `visualize_simulation_results.py` | Replot a saved NPZ (accuracy, excess delay, tradeoff, optional `eta`) |
| `visualize_r_sweep_tradeoff_excess_delay.py` | Aggregate an `--R_range` sweep into one accuracy-vs-excess-delay curve |

### Single-task example

```bash
python compare_to_baselines.py \
  --T 100 --seed 12 --c_min 10 --c_max 100 \
  --mu 6 --epsilon 0.1 --R 10 \
  --task toy_mlp_mnist --M 3 \
  --out_dir outputs/baseline_compare
```

### Multi-task example

```bash
python compare_to_baselines_multi.py \
  --T 150 --K 2 --c_min 100 --c_max 350 \
  --mu 10 --epsilon 0.1 \
  --w_k 1,2 --R_k 10,15 \
  --tasks toy_mlp_mnist,toy_mlp_mnist --M 3 \
  --out_dir outputs/baseline_compare_multi
```

Every time slot evaluates the true accuracy once per algorithm, so cost grows
with `--T`, with `--K`, and with the number of baselines. Lower `--T`, or pass
`--baseline_algorithms none`, while iterating.

### Replot saved results

```bash
python visualize_simulation_results.py \
  --npz outputs/baseline_compare/simulation_results.npz

python visualize_simulation_results.py \
  --npz outputs/baseline_compare_multi/simulation_results_multi.npz --plot_eta
```

### R-sweep trade-off plot

After a compare run with `--R_range` (outputs under `out_dir/R_<value>/`):

```bash
python compare_to_baselines.py --R_range 10,60,10 --out_dir outputs/baseline_R_sweep \
  --task toy_mlp_mnist --M 3

python visualize_r_sweep_tradeoff_excess_delay.py \
  --sweep_dir outputs/baseline_R_sweep
```

Registered task names are listed in `--task` / `--tasks` help text (via
`src.core.task_handler.registered_task_names()`).

## Package layout

```
src/core/                 # InferenceTask, callables, compressors, Stein oracles
src/core/resnet20.py      # CIFAR ResNet architectures (incl. ResNet-56)
src/core/task_instances/  # Plug-in task families
src/optimizers/           # CSI / no-CSI optimizers, estimators, baselines
profile_llm_stages.py     # Measures per-stage tau and per-link a for an LLM task
assets/                   # ResNet checkpoint, traces, fitting models, stage
                          #   profiles (see assets/README.md)
experiments/concavity/    # Concavity study: is utility concave in eta?
```

## Adding a task instance

1. Add `src/core/task_instances/<name>.py` with:
   - `setup_model_and_callables(**kwargs) -> (api, extra)`
   - `make_task(task_id, api, w_k=..., R_k=..., **kwargs) -> InferenceTask`
2. Register it in `src/core/task_handler.py` (or call `register_task(...)`).
3. Pass `--task <name>` (or `--tasks ...`) and set `--M` equal to the task's
   number of stages.

See the docstring in `src/core/task_instances/__init__.py` for the full contract.
Shared codecs live in `src/core/compressors.py`.

### Built-in tasks

| Name | Notes |
|------|--------|
| `toy_mlp_mnist` | Small MLP on MNIST (default demo; `--M 3`) |
| `resnet56_cifar10_topk` | ResNet-56 / CIFAR-10, top-`k` compression (`--M 4`) |
| `resnet56_cifar10_quantization` | Same module, quantization codec |
| `resnet56_cifar10_llmint8` | Same module, LLM.int8-style codec |

ResNet defaults load from `assets/` (checkpoint, 15 Mbps scenario trace, and
optional poly3 fitting models). Override paths with environment variables if
needed:

```bash
# defaults already point at assets/; CIFAR-10 downloads under ./data
python compare_to_baselines.py --task resnet56_cifar10_topk --M 4 \
  --c_min 1500000 --c_max 1500000 --T 100 --R 5 --out_dir outputs/resnet_compare

# optional overrides:
# export RESNET56_PHASE1_CHECKPOINT=/path/to/resnet56.th
# export RESNET56_PHASE1_TRACE_PATH=/path/to/scenario_trace.json
# export RESNET56_PHASE1_DATA_ROOT=./data
```

Other useful env vars: `RESNET56_PHASE1_DEVICE`, `RESNET56_PHASE1_FAST_SAMPLES`,
`RESNET56_PHASE1_STEIN_N`, `RESNET56_PHASE1_SIGMA`, `RESNET56_PHASE1_ETA_MIN`,
`RESNET56_PHASE1_DOWNLOAD` (set `0` on a node with no network),
`RESNET56_ACCURACY_ESTIMATOR_MODE` (`stein_estimator` or `fitting_model`), and
`RESNET56_FITTING_MODEL_PATH`.

## LLM tasks

The five language scenarios are task instances like any other, registered once
per codec. Append `_topk`, `_quantization` or `_llmint8` to the name, and set
`--M` to the number of stages: 5 for the Gemma and Llama tasks, 4 for Flan-T5.

| Task | Model | Data / metric | Scenario |
|---|---|---|---|
| `gemma2b_sharegpt_ppl` | Gemma-2B | ShareGPT, bounded perplexity | G1-2B-SGPT |
| `gemma7b_sharegpt_ppl` | Gemma-7B | ShareGPT, bounded perplexity | G1-7B-SGPT |
| `llama31_8b_wikitext_ppl` | Llama-3.1-8B | WikiText-2, bounded perplexity | Ll3-8B-WT |
| `llama31_8b_mmlu` | Llama-3.1-8B | 5-shot MMLU, accuracy | Ll3-8B-MMLU |
| `flant5_sst2` | Flan-T5-base | SST-2, accuracy | FT5-SST2 |

```bash
python compare_to_baselines.py \
  --task llama31_8b_wikitext_ppl_topk --M 5 \
  --T 50 --R 10 --c_min 4000000 --c_max 40000000 \
  --out_dir outputs/llama_wt_compare
```

The perplexity tasks report a bounded score,
`min{1, PPL_reference / PPL_compressed}`, so every task's utility lies in
`(0, 1]` like an accuracy.

Model weights are not bundled. The Gemma and Llama checkpoints are gated on the
Hub: run `hf auth login`, or point `HF_HOME` at a cache that has them.

`tau` and `a` are read from `assets/llm_stage_profiles.json`, keyed by
`(model, n_stages, batch_size, seq_len)`. The bundled profile covers every
task's defaults; if an entry is missing, setup fails with the command to run:

```bash
python profile_llm_stages.py --model google/gemma-2b --n_stages 5 \
    --batch_size 1 --seq_len 512
```

Optional per-task overrides: `LLM_TASK_N_STAGES`, `LLM_TASK_ETA_MIN`,
`LLM_TASK_FAST_SAMPLES`, `LLM_TASK_TRUE_SAMPLES`, `LLM_TASK_MAX_LENGTH`,
`LLM_TASK_BATCH_SIZE`, `LLM_TASK_GRAD_SIGMA`, `LLM_TASK_GRAD_N`,
`LLM_TASK_MMLU_N_SHOT`, `LLM_TASK_SEED`, and `FLANT5_SST2_DATA`. Changing
`LLM_TASK_N_STAGES`, `LLM_TASK_BATCH_SIZE` or `LLM_TASK_MAX_LENGTH` requires a
matching profile entry.

## Compression codecs

| Codec | `eta` selects |
|---|---|
| `topk` | the fraction of elements kept per token |
| `quantization` | a bit width from {2, 4, 8, 16, 32}, at a per-token scale |
| `llmint8` | a two-band split: FP16 outliers plus a low band stepping `int8` → `int4` → `int2` → drop |

For all three, `eta` is a compression ratio in `[0, 1]`, which is what makes
them comparable at equal `eta`. `COMPRESSOR_BACKEND=cpu|gpu` selects the
implementation; the two are not numerically identical, so fix it for the
duration of a campaign.

## Concavity study

`experiments/concavity/` measures whether task utility is concave in `eta`, the
property the optimizers rely on when they treat links semi-independently, across
all seven scenarios and three codecs. See
[experiments/concavity/README.md](experiments/concavity/README.md).

## Acknowledgements

This work was supported by the National Science Foundation through the AI-EDGE
Institute (Award No. 2112471), by the Army Research Laboratory under Grant
No. W911NF-24-2-0172, and by the Army Research Office under Grant No. W911NF-24-1-0103.

## Citation

```bibtex
@inproceedings{communication_aware_inference_2026,
  author    = {Peyman Gholami and
               Theodoros-Thirimachos Davarakis and
               Teng Li and
               Miquel {Sirera Perelló} and
               Salil Reddy and
               Ayberk Yarkın Yıldız and
               Anish Arora and
               Atilla Eryilmaz and
               Stratis Ioannidis and
               Chengzhang Li and
               Hulya Seferoglu and
               Ness Shroff},
  title     = {Communication-Aware Model Distributed Inference via
               Latent Representation Compression},
  booktitle = {Proceedings of ACM MobiHoc},
  year      = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).

This repository also redistributes third-party material under its own terms —
the CIFAR ResNet implementation and the pretrained ResNet-56 checkpoint are
BSD-2-Clause. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
