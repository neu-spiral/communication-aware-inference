#!/bin/bash
# =============================================================================
# run_mc_fill_chain.sh
#
# Chains 8h `mc_concavity.py fill` jobs via SLURM job-dependency (afterany) so
# a single task keeps resubmitting itself until all floors reach
# min_box_samples usable rays. Safe because cmd_fill resumes from
# <stem>_fill.partial.json and exits immediately ("All floors already have
# >= N usable rays... Skipping model load") once satisfied, so the final
# link(s) in the chain are cheap no-ops.
#
# Usage: bash experiments/concavity/slurm/run_mc_fill_chain.sh
# =============================================================================
set -e
# Resolve the repo root from this script's own location, so the scripts work
# from any cwd and carry no absolute paths. Override the conda env with
# CONDA_ENV=<name> if yours is not called "easy".
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-easy}"

HOME_DIR="${REPO_ROOT}"
MIN_BOX_SAMPLES=100
FLOOR_STEP=0.1
TIME="08:00:00"
MEM="48G"
MAX_LINKS=4   # safety cap on chain length per task

mkdir -p "${REPO_ROOT}/logs"

submit_chain() {
    local NAME=$1 MODEL=$2 DATASET=$3 METRIC=$4 STRATEGY=$5 RAYS_JSON=$6 \
          OUT_DIR=$7 TOL=$8 EXTRA=$9
    local JOBID=${FIRST_DEP:-}
    for i in $(seq 1 ${MAX_LINKS}); do
        local DEPARG=()
        if [ -n "${JOBID}" ]; then
            DEPARG=(--dependency=afterany:${JOBID})
        fi
        JOBID=$(sbatch --parsable \
            --partition=gpu --gres=gpu:a100:1 \
            --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=${MEM} --time=${TIME} \
            "${DEPARG[@]}" \
            --job-name=mcfillchain_${NAME} \
            --output=${REPO_ROOT}/logs/mcfillchain_${NAME}_link${i}_%j.out \
            --error=${REPO_ROOT}/logs/mcfillchain_${NAME}_link${i}_%j.err \
            --wrap="
set -e
echo \"[wrap] link ${i} Job \$SLURM_JOB_ID on \$(hostname) at \$(date)\" >&2
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate ${CONDA_ENV}
python ${HOME_DIR}/experiments/concavity/mc_concavity.py fill \\
    --rays_json ${RAYS_JSON} \\
    --model ${MODEL} --dataset ${DATASET} --metric ${METRIC} --strategy ${STRATEGY} \\
    --tol ${TOL} --min_box_samples ${MIN_BOX_SAMPLES} --adaptive_floor_step ${FLOOR_STEP} \\
    ${EXTRA} \\
    --out_dir ${OUT_DIR}")
        echo "  link ${i}: ${NAME} -> job ${JOBID}"
    done
}

PPL_EXTRA="--max_texts 128 --max_length 512 --batch_size 1"
ACC_EXTRA="--samples_per_subject 20 --n_shot 5 --max_length 512 --batch_size 16"

OUT="${REPO_ROOT}/outputs/mc_concavity"

# FIRST_DEP: chain after the currently-running job (8042521) so we don't
# duplicate GPU work on the same rays file while it's still writing.
echo "Chaining fill for: llama_mmlu_reserve_c4 (currently running as 8042521, will continue after it exits)"
FIRST_DEP=8042521 submit_chain llama_mmlu_reserve_c4 meta-llama/Llama-3.1-8B mmlu accuracy llmint8_reserve \
    ${OUT}/meta-llama_Llama-3.1-8B/mmlu_accuracy/reserve_merged_rays.json \
    ${OUT}/meta-llama_Llama-3.1-8B/mmlu_accuracy 0.02 "${ACC_EXTRA}"

echo "Done submitting chain."
