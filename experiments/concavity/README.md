# Concavity experiments

Measures whether task utility is concave in the per-link compression ratio
vector `eta`. The optimizers in `src/optimizers/` solve each link's ratio
semi-independently, which is sound only under concavity.

Concavity is not uniform over the domain: it holds where `eta` is bounded away
from zero and degrades as the bound drops. The results are therefore a sweep of
the fraction of Jensen's-inequality checks that pass against an `eta` floor `c`,
at several chord tolerances, for each compression method and task.

## The test

Along a segment through `eta`-space the task metric is evaluated at several
points. For points `i < j < k` on the segment, `j` is a convex combination of
the endpoints:

```
t_j = lam * t_i + (1 - lam) * t_k        lam = (t_k - t_j) / (t_k - t_i)
```

Concavity requires the middle value to lie on or above the chord:

```
f(j) >= lam * f(i) + (1 - lam) * f(k) - tol
```

The **Jensen inequality pass rate** is the fraction of sampled triples that
satisfy this, and is the y axis of every figure here
(`jensen_concavity.Y_AXIS_LABEL`).

`tol` absorbs evaluation noise. For a discrete metric that noise is
one-directional, so a `tol = 0` panel is a lower bound rather than the truth;
this is why the figures sweep tolerance instead of reporting one number.

At floor `c` only the portion of each segment inside `[c, 1]^n` counts, and a
segment contributing fewer than three in-floor points is discarded whole.

## Figures

`figures/` holds one figure per compression method:

| File | Method |
|---|---|
| `concavity_grid_topk.{png,pdf}` | top-k per token |
| `concavity_grid_llmint8.{png,pdf}` | LLM.int8()-style mixed precision |
| `concavity_grid_quantization.{png,pdf}` | uniform quantization |

Each figure has one panel per tolerance in a single row, left to right as the
test loosens, and one line per task. Colour and marker are keyed to task
identity, so a task keeps its appearance across all three figures. Figures
carry no title; the method is in the filename.

Lines are labelled with the paper's scenario names and stage count `L_k`:

| Curve | Model / data / metric | On-disk key |
|---|---|---|
| `G1-2B-SGPT (L_k=5)` | gemma-2b, ShareGPT, perplexity | `cuts4` |
| `G1-2B-SGPT (L_k=8)` | gemma-2b, ShareGPT, perplexity | `cuts7` |
| `G1-7B-SGPT (L_k=5)` | gemma-7b, ShareGPT, perplexity | `cuts4` |
| `Ll3-8B-WT (L_k=5)` | Llama-3.1-8B, WikiText-2, perplexity | `cuts4` |
| `Ll3-8B-MMLU (L_k=5)` | Llama-3.1-8B, MMLU, accuracy | `cuts4` |
| `FT5-SST2 (L_k=4)` | flan-t5-base, SST-2, accuracy | `cuts3` |
| `RN56-CF10 (L_k=4)` | ResNet-56, CIFAR-10, accuracy | `cuts3` |

Two naming conventions apply throughout:

- **`n_cuts` counts links, `L_k` counts stages.** Paths, filenames and JSON keys
  use `cuts<N>`, where `N` is the number of compressed links, i.e. `dim(eta)`.
  An `L_k`-stage path has `L_k - 1` links, so plots display `L_k = n_cuts + 1`.
  The conversion belongs in `plot_concavity_grid.task_label`; on-disk keys stay
  as they are.
- **The ResNet row is CIFAR-10** but is keyed `dataset="imagenet"`. Only the
  display name is corrected, in `jensen_concavity._DATASET_DISPLAY`. Renaming
  the key splits the group and halves its ray pool.

## Layout

```
experiments/concavity/
  README.md
  figures/                        released figures (png + pdf)

  jensen_concavity.py             floor sweep, one figure per task
  plot_concavity_grid.py          ray-pool loader, task labels, combined grid
  plot_concavity_per_method.py    produces the released figures

  mc_concavity.py                 LLM Monte-Carlo rays: run / fill / etamin
  ray_concavity_score.py          fixed-ray test; also supplies the corpus and
                                  scoring helpers mc_concavity reuses
  rays_interesting*.json          hand-designed rays for the fixed-ray test
  enumerate_quant_concavity.py    quantization: exhaustive lattice enumeration
  quant_box_rays.py               quantization: random rays read off the lattice
  merge_quant_shards.py           merges sharded quantization runs
  resnet_mc_concavity.py          ResNet-56 / CIFAR-10 arm
  flant5_sst2/                    Flan-T5 / SST-2 harness
  slurm/                          submit scripts for the long runs
```

Ray data and generated plots land under `outputs/`, which is gitignored.
`figures/` is committed because those files are the released artifacts.

## Requirements

Plotting needs `numpy` and `matplotlib`. Data generation also needs `torch`,
`transformers`, a GPU, and local copies of the gated Llama and Gemma
checkpoints. Submit scripts activate a conda env named `easy`; override with
`CONDA_ENV=<name>`, or edit the two `conda` lines in the wrapper if you followed
the venv install in the top-level README. `run_quant_enum.sh` and
`run_t5_llmint8.sh` additionally export `HF_HOME` from `HF_CACHE`, which
defaults to `${HOME}/.cache/huggingface`; point it at a cache that already holds
the gated checkpoints to run offline.

Corpus and MMLU loading fetches Hub parquet directly through
`huggingface_hub` and reads it with pandas, because the pinned
`datasets`/`fsspec` pair cannot resolve the Hub's glob patterns.
`datasets.load_dataset` remains as a fallback.

## Reproducing the figures

`outputs/` is gitignored, so the ray tree is **not** shipped with the
repository: a fresh clone has `figures/` (the released artifacts) but no
`outputs/mc_concavity/`. Regenerate the rays first with the four arms under
[Regenerating ray data](#regenerating-ray-data) — that is the part that needs a
GPU and the gated checkpoints — or ask the authors for the ray tree.

With `outputs/mc_concavity/` populated, the plotting step itself needs no model
and no GPU:

```bash
python experiments/concavity/plot_concavity_per_method.py \
    --root        outputs/mc_concavity \
    --resnet_root outputs/concavity_resnet \
    --out_dir     outputs/mc_concavity/jensen
```

The script discovers every `*_rays.json` and `*.partial.json` under `--root`,
groups rays by `(model, dataset, metric, n_cuts, strategy)`, de-duplicates by
content hash so repeated seeds, phases and checkpoints merge without
double-counting, and writes the three figures plus `concavity_per_method.json`,
which holds the values behind every plotted line. Output is deterministic for a
given ray tree and `--seed`.

| Flag | Default | Effect |
|---|---|---|
| `--tols` | `0.0 0.01 0.02 0.05` | one panel per tolerance. `0.1` is excluded: every task saturates at 1.0 there |
| `--floor_max` | `0.7` | adaptive fill guarantees usable rays only to 0.7 |
| `--samples_per_ray` | `5` | triples drawn per ray per floor, per repeat |
| `--n_repeats` | `5` | independent redraws; the band is ±1 std across them |
| `--exclude` | `flan-t5-base:ppl_score` | the T5 perplexity-score variant has no paper scenario |

The shaded band is the spread over repeated small triple draws, so it measures
sensitivity to *which* triples were sampled. `jensen_concavity.py` instead
samples many more triples per ray and shades ±1 binomial standard error.

Two other views:

```bash
# one figure per task, one curve per strategy, +/-1 binomial SE
python experiments/concavity/jensen_concavity.py \
    --root outputs/mc_concavity --out_dir outputs/mc_concavity/jensen

# all tasks in one grid, one line per (strategy, tol)
python experiments/concavity/plot_concavity_grid.py \
    --root outputs/mc_concavity --out outputs/mc_concavity/jensen/concavity_grid.png
```

## Regenerating ray data

Four arms write the same ray schema and can be regenerated independently.

### LLM Monte-Carlo rays (gemma, Llama)

```bash
# smoke test. Phase 2 still defaults to 80 rays, so this is 100 rays in total:
# measured 773 s on a V100-SXM2 at 7.4 s/ray, model load included. Add
# --phase2_rays 0 for a phase-1-only version in about a fifth of that.
python experiments/concavity/mc_concavity.py run \
    --model meta-llama/Llama-3.1-8B \
    --dataset wikitext --metric perplexity --strategy topk_per_token \
    --n_cuts 4 --n_rays 20 --n_points 7 --max_texts 16 \
    --min_box_samples 0 \
    --out_dir outputs/mc_concavity/smoke

# full task matrix, one SLURM job per (model, dataset, strategy)
bash experiments/concavity/slurm/run_mc_concavity.sh
```

Three phases run in sequence: `n_rays` isotropic random rays over `[eps, 1]^n`;
`phase2_rays` targeted rays inside the `eta_min` candidate sub-cube; then an
adaptive fill that samples until every floor level in `[0, 0.9]` has
`min_box_samples` whole rays with at least three in-floor points. A ray sampled
at floor `c` also counts for every lower floor.

Progress checkpoints to `*.partial.json`, and the plot scripts read partials, so
a job stopped by a walltime limit still contributes. Two subcommands build on
that:

```bash
# top up an existing tree; exits before loading the model if all floors are full
python experiments/concavity/mc_concavity.py fill --rays_json <...>_rays.json ...

# re-run the eta_min search on saved rays, no model required
python experiments/concavity/mc_concavity.py etamin --rays_json <...>_rays.json \
    --tol 0.02 --target 0.95 --min_usable 30
```

An 8-hour job covers roughly a 150-ray MMLU run at `n_points=9`. `llmint8` is
about 3x slower per ray than `topk_per_token` and usually needs chaining:
`slurm/run_mc_fill_chain.sh` submits a `--dependency=afterany` chain that
resumes from the checkpoint until the floors fill.

| Metric | `n_rays` | `n_points` | `tol` |
|---|---|---|---|
| perplexity (WikiText, ShareGPT) | 150 | 7-9 | 0.001 |
| accuracy (MMLU) | 150-200 | 9 | 0.02 |

### Quantization

Quantization `eta` snaps to a bit-width rung, so the reachable points form a
lattice and the whole grid is enumerated rather than sampled.

```bash
bash experiments/concavity/slurm/run_quant_enum.sh          # sharded enumeration
python experiments/concavity/merge_quant_shards.py --help   # merge the shards
```

`enumerate_quant_concavity.py` walks the `5^n` grid and emits axis-aligned rays.
`quant_box_rays.py` then reads random rays off that lattice without further
evaluation, which is what makes the quantization curve comparable to the
continuously-parameterised methods.

- **Requested versus delivered `eta`.** The floor constraint `eta_i >= c`
  applies to the requested ratio; snapping is what the backend delivers. Ray
  files hold the requested vector in `"eta"` and the lattice coordinates in
  `"eta_snapped"`. This is also what lets the sweep reach `c = 0.7`: above
  `c = 0.5` every coordinate snaps to the same rung, so the ray is flat and
  trivially concave rather than discarded. Flat rays require the numerically
  stable chord form `y_k + lam * (y_i - y_k)` to pass at `tol = 0`, which both
  chord implementations use.
- **Snap rule.** `--snap floor` applies one rule to all groups. The ResNet GPU
  dispatcher ceils, so for ResNet `floor` is an analysis convention rather than
  a description of the backend; `--snap code` reproduces per-backend behaviour.

### ResNet-56 / CIFAR-10

```bash
bash experiments/concavity/slurm/run_resnet_mc_fill.sh
```

`resnet_mc_concavity.py` drives `src.core.resnet_task_callables` and needs the
ResNet-56 checkpoint (`assets/resnet56-4bfd9763.th`) and a CIFAR-10 root
(`--data_root`).

### Flan-T5 / SST-2

```bash
bash experiments/concavity/slurm/run_t5_llmint8.sh
```

`flant5_sst2/` is a self-contained harness: encoder-decoder models need
different hooks and scoring from the causal-LM path. Its `compressors.py`
quantizes with per-row scales, where `src/core/compressors.py` uses one
per-tensor scale, so the two are not interchangeable. llmint8 parameters come
from `src.core.llmint8`, the single definition of that scheme.

The SST-2 validation split is committed at
`flant5_sst2/data/sst2_validation.jsonl`, so the harness runs without Hub
access.

## Compressor backends

`COMPRESSOR_BACKEND=cpu|gpu` selects between two quantizer implementations that
are **not** numerically identical:

| Implementation | Quantizer scale |
|---|---|
| `src/core/compressors.py` | one per-tensor abs-max |
| `src/core/gpu_compressors.py` | per-row (per-token) abs-max |

A per-row scale is strictly finer, so the GPU path has lower round-trip error
(measured RMSE ratios 0.30-0.53). `tests/test_gpu_compressors.py` pins that
relationship. Fix the backend for the duration of a campaign: switching it
changes measured utilities.

`eta` has one meaning for every method here, a compression ratio in `[0, 1]`,
which is what makes the three figures comparable at equal `eta`. See
`src/core/llmint8.py` for the bit allocation it implies for llmint8.
