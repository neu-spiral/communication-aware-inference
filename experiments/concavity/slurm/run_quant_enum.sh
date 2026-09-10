#!/bin/bash
# =============================================================================
# run_quant_enum.sh
#
# Exhaustive QUANTIZATION concavity enumeration for the causal-LM groups
# (gemma / llama) via experiments/concavity/enumerate_quant_concavity.py.
#
# Quantization snaps each cut's eta to the paper ladder {0.0625,0.125,0.25,0.5,1.0}
# = {2,4,8,16,32}-bit (FP32 native), so instead of Monte-Carlo random rays we
# evaluate every L**n = 5**n lattice point EXACTLY and express the grid as
# axis-aligned straight-line rays that jensen_concavity.py folds in unchanged.
#
# Models load in FP32 (--dtype fp32): the paper's native reference is 32-bit, so
# eta=1 is a true uncompressed operating point. Large FP32 models are sharded
# across 2 A100s with device_map=auto.
#
# Grid sizes: cuts3 -> 125, cuts4 -> 625 (both fit one 8h link); cuts7 -> 78125
# (chained: enumerate_quant_concavity.py resumes from <stem>_grid.partial.json,
# so afterany links keep topping up until the grid is complete, then become
# cheap no-ops). Re-run this script to add more cuts7 waves if needed.
#
# Usage:
#   bash experiments/concavity/slurm/run_quant_enum.sh                 # first batch (see TASKS)
#   TASKS=all bash experiments/concavity/slurm/run_quant_enum.sh       # everything
#   TASKS="gemma2b_c7" C7_LINKS=6 bash experiments/concavity/slurm/run_quant_enum.sh
# =============================================================================
set -e
# Resolve the repo root from this script's own location, so the scripts work
# from any cwd and carry no absolute paths. Override the conda env with
# CONDA_ENV=<name> if yours is not called "easy".
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-easy}"

REPO="${REPO_ROOT}"
TIME="${TIME:-08:00:00}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
C7_LINKS="${C7_LINKS:-4}"          # afterany links for the cuts7 (78k-point) grid
OUT="${REPO}/outputs/mc_concavity"
mkdir -p "${REPO_ROOT}/logs"

PPL_EXTRA="--max_texts 128 --max_length 512 --batch_size 1"
ACC_EXTRA="--samples_per_subject 20 --n_shot 5 --max_length 512 --batch_size 8"

# submit_enum NAME MODEL DATASET METRIC NCUTS OUT_DIR EXTRA GRES MEM LINKS
submit_enum() {
    local NAME=$1 MODEL=$2 DATASET=$3 METRIC=$4 NCUTS=$5 OUT_DIR=$6 \
          EXTRA=$7 GRES=$8 MEM=$9 LINKS=${10}
    mkdir -p "${OUT_DIR}"
    local JOBID=""
    for i in $(seq 1 ${LINKS}); do
        local DEPARG=()
        [ -n "${JOBID}" ] && DEPARG=(--dependency=afterany:${JOBID})
        JOBID=$(sbatch --parsable \
            --partition=gpu --gres=${GRES} \
            --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=${MEM} --time=${TIME} \
            "${DEPARG[@]}" \
            --job-name=qenum_${NAME} \
            --output="${REPO_ROOT}/logs/qenum_${NAME}_link${i}_%j.out" \
            --error="${REPO_ROOT}/logs/qenum_${NAME}_link${i}_%j.err" \
            --wrap="
set -e
echo \"[wrap] link ${i} Job \$SLURM_JOB_ID on \$(hostname) at \$(date)\" >&2
export HF_HOME=${HF_CACHE}
export HF_TOKEN=\$(cat ${HF_CACHE}/token 2>/dev/null)
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate ${CONDA_ENV}
cd ${REPO}
python experiments/concavity/enumerate_quant_concavity.py \\
    --model ${MODEL} --dataset ${DATASET} --metric ${METRIC} \\
    --n_cuts ${NCUTS} --dtype fp32 ${EXTRA} \\
    --out_dir ${OUT_DIR}")
        echo "  link ${i}: ${NAME} -> job ${JOBID}"
    done
}

TASKS="${TASKS:-gemma2b_c4 gemma7b_c4 llama_mmlu_c4 llama_wikitext_c4}"
want() { [ "${TASKS}" = "all" ] && return 0; for t in ${TASKS}; do [ "$t" = "$1" ] && return 0; done; return 1; }

# gemma-2b FP32 (~10GB) fits one A100; gemma-7b/llama FP32 sharded across 2.
want gemma2b_c4 && submit_enum gemma2b_c4 google/gemma-2b sharegpt perplexity 4 \
    "${OUT}/google_gemma-2b/sharegpt_perplexity/quantization/cuts4" \
    "${PPL_EXTRA}" gpu:a100:1 48G 1

want gemma2b_c7 && submit_enum gemma2b_c7 google/gemma-2b sharegpt perplexity 7 \
    "${OUT}/google_gemma-2b/sharegpt_perplexity/quantization/cuts7" \
    "${PPL_EXTRA}" gpu:a100:1 48G ${C7_LINKS}

want gemma7b_c4 && submit_enum gemma7b_c4 google/gemma-7b sharegpt perplexity 4 \
    "${OUT}/google_gemma-7b/sharegpt_perplexity/quantization/cuts4" \
    "${PPL_EXTRA}" gpu:a100:2 96G 1

want llama_mmlu_c4 && submit_enum llama_mmlu_c4 meta-llama/Llama-3.1-8B mmlu accuracy 4 \
    "${OUT}/meta-llama_Llama-3.1-8B/mmlu_accuracy/quantization/cuts4" \
    "${ACC_EXTRA}" gpu:a100:2 96G 1

want llama_wikitext_c4 && submit_enum llama_wikitext_c4 meta-llama/Llama-3.1-8B wikitext perplexity 4 \
    "${OUT}/meta-llama_Llama-3.1-8B/wikitext_perplexity/quantization/cuts4" \
    "${PPL_EXTRA}" gpu:a100:2 96G 1

echo "Done submitting. (TASKS=${TASKS})"
