#!/bin/bash
# =============================================================================
# run_mc_concavity.sh
#
# Monte-Carlo concavity estimation in eta-space + per-cut-point eta_min search.
# Reframes the fixed-ray test (run_ray_concavity_score.sh): instead of a few
# hand-designed rays, we sample many RANDOM rays in the eta-hypercube [0,1]^n,
# evaluate the task metric along each, and report concavity = fraction of rays
# that are concave. Ray profiles are saved so the per-cut-point eta_min vector
# (smallest sub-cube [eta_min,1]^n with concavity >= 95%) can be searched.
#
# Task matrix (causal-LM, this pass):
#   gemma-2b   ShareGPT  perplexity   n_cuts=4  AND 7      (cached base model)
#   gemma-7b   ShareGPT  perplexity   n_cuts=4
#   Llama-3.1-8B  MMLU      accuracy     n_cuts=4
#   Llama-3.1-8B  WikiText  perplexity   n_cuts=4
# Strategies (both): topk_per_token, llmint8_reserve
#
# One SLURM job per (model, dataset/metric, strategy). Each job sweeps its
# configured n_cuts list sequentially into cuts<N>/ subdirs.
#
# Usage:  bash experiments/concavity/slurm/run_mc_concavity.sh
# =============================================================================

# Resolve the repo root from this script's own location, so the scripts work
# from any cwd and carry no absolute paths. Override the conda env with
# CONDA_ENV=<name> if yours is not called "easy".
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-easy}"

HOME_DIR="${REPO_ROOT}"

# ── Monte-Carlo parameters ───────────────────────────────────────────────
N_RAYS=150          # Phase-1 random rays over the whole hypercube
N_POINTS=7          # evaluation points along each ray (t in [0,1])
PHASE2_RAYS=80      # Phase-2 targeted rays inside the candidate sub-cube (0=skip)
TARGET=0.95         # required concavity fraction for eta_min
MIN_USABLE=30       # min clipped rays needed to trust a sub-cube estimate

# perplexity eval
MAX_TEXTS=128
MAX_LENGTH=512
BATCH_SIZE=1
PPL_TOL=0.001       # D2 concavity tolerance for perplexity (smooth) tasks

# mmlu eval (accuracy is discrete/noisy -> larger tol, more samples)
SAMPLES_PER_SUBJECT=20
N_SHOT=5
ACC_TOL=0.02

STRATEGIES=("topk_per_token" "llmint8_reserve")

TIME="08:00:00"
MEM="48G"
OUT_BASE="${HOME_DIR}/outputs/mc_concavity"
# =============================================================================

mkdir -p "${REPO_ROOT}/logs"

# submit MODEL DATASET METRIC TOL EXTRA_ARGS NCUTS_CSV
submit() {
    local MODEL=$1 DATASET=$2 METRIC=$3 TOL=$4 EXTRA=$5 NCUTS_CSV=$6
    local MODEL_SLUG; MODEL_SLUG=$(echo "${MODEL}" | tr '/' '_')

    local PY_CMDS=""
    IFS=',' read -ra NCUTS <<< "${NCUTS_CSV}"
    for N in "${NCUTS[@]}"; do
        local OUT_DIR="${OUT_BASE}/${MODEL_SLUG}/${DATASET}_${METRIC}/${STRATEGY}/cuts${N}"
        PY_CMDS="${PY_CMDS}
mkdir -p ${OUT_DIR}
python ${HOME_DIR}/experiments/concavity/mc_concavity.py run \\
    --model ${MODEL} --dataset ${DATASET} --metric ${METRIC} \\
    --strategy ${STRATEGY} --n_cuts ${N} \\
    --n_rays ${N_RAYS} --n_points ${N_POINTS} --phase2_rays ${PHASE2_RAYS} \\
    --target ${TARGET} --min_usable ${MIN_USABLE} --tol ${TOL} \\
    ${EXTRA} \\
    --out_dir ${OUT_DIR}
"
    done

    echo "Submitting: ${MODEL} ${DATASET}/${METRIC} ${STRATEGY} cuts={${NCUTS_CSV}}"
    sbatch --parsable \
        --partition=gpu --gres=gpu:a100:1 \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=${MEM} --time=${TIME} \
        --job-name=mcconc_${MODEL_SLUG}_${DATASET}_${STRATEGY} \
        --output=${REPO_ROOT}/logs/mcconc_${MODEL_SLUG}_${DATASET}_${STRATEGY}_%j.out \
        --error=${REPO_ROOT}/logs/mcconc_${MODEL_SLUG}_${DATASET}_${STRATEGY}_%j.err \
        --wrap="
set -e
echo \"[wrap] Job \$SLURM_JOB_ID on \$(hostname) at \$(date)\" >&2
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate ${CONDA_ENV}
${PY_CMDS}"
}

PPL_EXTRA="--max_texts ${MAX_TEXTS} --max_length ${MAX_LENGTH} --batch_size ${BATCH_SIZE}"
ACC_EXTRA="--samples_per_subject ${SAMPLES_PER_SUBJECT} --n_shot ${N_SHOT} --max_length ${MAX_LENGTH} --batch_size 8"

for STRATEGY in "${STRATEGIES[@]}"; do
    submit "google/gemma-2b"          sharegpt perplexity "${PPL_TOL}" "${PPL_EXTRA}" "4,7"
    submit "google/gemma-7b"          sharegpt perplexity "${PPL_TOL}" "${PPL_EXTRA}" "4"
    submit "meta-llama/Llama-3.1-8B"  mmlu     accuracy   "${ACC_TOL}" "${ACC_EXTRA}" "4"
    submit "meta-llama/Llama-3.1-8B"  wikitext perplexity "${PPL_TOL}" "${PPL_EXTRA}" "4"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Per-run outputs under ${OUT_BASE}/<model>/<dataset>_<metric>/<strategy>/cuts<N>/:"
echo "  *_rays.json     full ray profiles (eta-vectors + metric values)"
echo "  *_summary.json  global concavity fraction + eta_min (phase1/phase2)"
echo "  *_rays.png      ray curves (green=concave, red=non-concave)"
echo ""
echo "Re-run eta_min search model-free, e.g.:"
echo "  python ${HOME_DIR}/experiments/concavity/mc_concavity.py etamin --rays_json <…_rays.json>"
