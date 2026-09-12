# Concavity experiments

Measures whether task utility is concave in the per-link compression ratio
vector `eta`. The optimizers in `src/optimizers/` solve each link's ratio
semi-independently, which is sound only under concavity.

Concavity is not uniform over the domain: it holds where `eta` is bounded away
from zero and degrades as the bound drops. The result is therefore a sweep of
the fraction of Jensen's-inequality checks that pass against an `eta` floor `c`,
at several chord tolerances, for each compression method and task.

## The test

The task metric is evaluated at several points along a segment through
`eta`-space. For points `i < j < k` on the segment, `j` is a convex combination
of the endpoints with `lam = (t_k - t_j) / (t_k - t_i)`, and concavity requires
the middle value to lie on or above the chord:

```
f(j) >= lam * f(i) + (1 - lam) * f(k) - tol
```

The **Jensen inequality pass rate** is the fraction of sampled triples that
satisfy this. `tol` absorbs evaluation noise, which is why the figures sweep
tolerance instead of reporting one number. At floor `c` only the portion of each
segment inside `[c, 1]^n` counts.

## Figures

`figures/` holds the released figures, one per compression method:

| File | Method |
|---|---|
| `concavity_grid_topk.{png,pdf}` | top-k per token |
| `concavity_grid_llmint8.{png,pdf}` | LLM.int8()-style mixed precision |
| `concavity_grid_quantization.{png,pdf}` | uniform quantization |

Each has one panel per tolerance, left to right as the test loosens, and one
line per task, labelled with the paper's scenario names and stage count `L_k`:
`G1-2B-SGPT`, `G1-7B-SGPT`, `Ll3-8B-WT`, `Ll3-8B-MMLU`, `FT5-SST2`,
`RN56-CF10`. On disk the same groups are keyed `cuts<N>`, where `N` is the
number of compressed links, so `L_k = N + 1`.

## Regenerating the ray data

Ray data lands under `outputs/`, which is gitignored: a fresh clone has the
figures but not the rays behind them. Each backend below writes the same ray
schema and can be run on its own. Plotting needs only `numpy` and `matplotlib`;
generating rays needs `torch`, `transformers`, a GPU, and local copies of the
gated Llama and Gemma checkpoints.

### Gemma / Llama — top-k and LLM.int8

```bash
python experiments/concavity/mc_concavity.py run \
    --model meta-llama/Llama-3.1-8B \
    --dataset wikitext --metric perplexity \
    --strategy topk_per_token --n_cuts 4 \
    --out_dir outputs/mc_concavity
```

`--strategy` is `topk_per_token` or `llmint8_reserve`, `--dataset` is
`wikitext`, `sharegpt` or `mmlu`. Random rays are sampled over `[eps, 1]^n` and
an adaptive pass then fills the higher `eta` floors; `mc_concavity.py fill
--rays_json <...>_rays.json` tops up an existing tree, and `etamin` re-runs the
`eta_min` search on saved rays without loading a model.

### Gemma / Llama — quantization

Quantization snaps each `eta` to a bit-width rung, so the reachable points form
a lattice: it is enumerated once, and rays are then read off it without
evaluating the model again.

```bash
python experiments/concavity/enumerate_quant_concavity.py \
    --model meta-llama/Llama-3.1-8B \
    --dataset wikitext --metric perplexity --n_cuts 4 \
    --out_dir outputs/mc_concavity

python experiments/concavity/quant_box_rays.py --all
```

### ResNet-56 / CIFAR-10

```bash
python experiments/concavity/resnet_mc_concavity.py \
    --compressor topk --out_dir outputs/mc_concavity/resnet/topk
```

`--compressor` is `topk`, `quantization` or `llmint8`. Needs the ResNet-56
checkpoint (`assets/resnet56-4bfd9763.th`) and a CIFAR-10 root (`--data_root`).
These rays are keyed `dataset="imagenet"` for historical reasons; only the
display name is corrected, in `jensen_concavity._DATASET_DISPLAY`.

### Flan-T5 / SST-2

```bash
python experiments/concavity/flant5_sst2/flan_t5_sst2_concavity.py run \
    --compressor_name topk \
    --out_dir outputs/mc_concavity/flan-t5-base/topk
```

`flant5_sst2/` is a self-contained harness, because encoder-decoder models need
different hooks and scoring from the causal-LM path. The SST-2 validation split
is committed at `flant5_sst2/data/sst2_validation.jsonl`, so it runs without Hub
access.

## Redrawing the figures

Once `outputs/mc_concavity/` is populated, plotting needs no model and no GPU.
The quantization curves come from the box-ray root, so build that first:

```bash
python experiments/concavity/build_boxray_root.py --out outputs/_analysis_boxrays

python experiments/concavity/plot_concavity_per_method.py \
    --root    outputs/_analysis_boxrays \
    --out_dir outputs/mc_concavity/jensen
```

The script discovers every `*_rays.json` and `*.partial.json` under `--root`,
groups rays by `(model, dataset, metric, n_cuts, strategy)`, de-duplicates by
content hash, and writes the three figures plus `concavity_per_method.json`,
which holds the values behind every plotted line. Two other views of the same
pool: `jensen_concavity.py` (one figure per task, one curve per strategy) and
`plot_concavity_grid.py` (all tasks in one grid).
