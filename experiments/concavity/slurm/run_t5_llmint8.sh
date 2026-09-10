#!/bin/bash
# =============================================================================
# run_t5_llmint8.sh  (2026-07-20)
#
# LLM.int8() variant of the Flan-T5-base / SST-2 concavity campaign. T5 was
# topk-ONLY until now; this adds the missing llmint8 arm so T5 appears in the
# llmint8 per-method grid alongside gemma/llama/resnet.
#
# Mirrors the original topk T5 run EXACTLY (from its summary.json):
#   n_rays 200 (phase1) + phase2_rays 40, n_points 7, cuts 3, seed 0,
#   tol accuracy 0.01 / ppl_score 0.001, max_input_length 128, max_samples 0
#   (full 872-example validation split), dataset via the local jsonl fixture.
# llmint8 precision uses the harness defaults (outlier fp16 / regular int8),
# which is the same "reserve" scheme used for the gemma/llama llmint8_reserve
# arms, so the curves are comparable.
#
# One `run` evaluates BOTH metrics in a single model load, writing flat into a
# staging dir; we then move each metric's files into the nested
#   outputs/mc_concavity/google__flan-t5-base/sst2_<metric>/llmint8/cuts3/
# layout that jensen_concavity.py / check_ray_coverage.py --discover expect
# (discovery is by filename metadata, but we keep the layout consistent with
# the topk arm). After the run, a 3-link afterany fill chain per metric tops up
# the high-eta-floor sub-cubes to >= 100 usable rays up to floor 0.7, exactly
# like run_fill_t5.sh did for topk.
#
# flan-t5-base is already cached in the scratch HF cache; compute nodes have
# network if a re-fetch is needed.
#
# Usage: bash experiments/concavity/slurm/run_t5_llmint8.sh
# =============================================================================
set -e
# Resolve the repo root from this script's own location, so the scripts work
# from any cwd and carry no absolute paths. Override the conda env with
# CONDA_ENV=<name> if yours is not called "easy".
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-easy}"

HOME_DIR="${REPO_ROOT}"
MIN=100
FLOOR_STEP=0.1
FLOOR_MAX=0.7
TIME="08:00:00"
MEM="48G"
MAX_LINKS=3
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
mkdir -p "${REPO_ROOT}/logs"

T5=experiments/concavity/flant5_sst2/flan_t5_sst2_concavity.py
DATASET=flant5_sst2_concavity_test/data/sst2_validation.jsonl
MODEL_SLUG=google__flan-t5-base
BASE=outputs/mc_concavity/google__flan-t5-base
STAGE=${BASE}/_llmint8_run
ACC_DIR=${BASE}/sst2_accuracy/llmint8/cuts3
PPL_DIR=${BASE}/sst2_ppl_score/llmint8/cuts3
COMMON="--dataset_path ${DATASET} --n_points 7 --max_input_length 128 --max_samples 0 --seed 0"

mkdir -p "${STAGE}" "${ACC_DIR}" "${PPL_DIR}"

# ---- Link 0: the MC run (phase1 + phase2), both metrics, then stage move -----
RUN_JOB=$(sbatch --parsable \
    --partition=gpu --gres=gpu:a100:1 \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=${MEM} --time=${TIME} \
    --job-name=t5llmint8_run \
    --output=${REPO_ROOT}/logs/t5llmint8_run_%j.out \
    --error=${REPO_ROOT}/logs/t5llmint8_run_%j.err \
    --wrap="
set -e
echo \"[wrap] t5 llmint8 run Job \$SLURM_JOB_ID on \$(hostname) at \$(date)\" >&2
export HF_HOME=${HF_CACHE}
export HF_TOKEN=\$(cat ${HF_CACHE}/token 2>/dev/null)
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate ${CONDA_ENV}
python ${HOME_DIR}/${T5} run \\
    --compressor_name llmint8 \\
    --metrics accuracy ppl_score \\
    --n_rays 200 --phase2_rays 40 \\
    --tol_accuracy 0.01 --tol_ppl_score 0.001 \\
    --out_dir ${STAGE} \\
    ${COMMON}
# --- move each metric's artifacts into the nested campaign layout ---
mv -f ${STAGE}/${MODEL_SLUG}__sst2__accuracy__llmint8__cuts3* ${ACC_DIR}/
mv -f ${STAGE}/${MODEL_SLUG}__sst2__ppl_score__llmint8__cuts3* ${PPL_DIR}/
echo \"[wrap] staged llmint8 rays into ${ACC_DIR} and ${PPL_DIR}\" >&2
")
echo "run link 0: t5llmint8_run -> job ${RUN_JOB}"

# ---- Fill chain (per metric), each afterany on the previous link ------------
submit_fill_chain() {
    local NAME=$1 RAYS_JSON=$2 OUT_DIR=$3
    local JOBID=${RUN_JOB}   # first fill link waits on the run job
    for i in $(seq 1 ${MAX_LINKS}); do
        JOBID=$(sbatch --parsable \
            --partition=gpu --gres=gpu:a100:1 \
            --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=${MEM} --time=${TIME} \
            --dependency=afterany:${JOBID} \
            --job-name=fillt5i8_${NAME} \
            --output=${REPO_ROOT}/logs/fillt5i8_${NAME}_link${i}_%j.out \
            --error=${REPO_ROOT}/logs/fillt5i8_${NAME}_link${i}_%j.err \
            --wrap="
set -e
echo \"[wrap] fill link ${i} Job \$SLURM_JOB_ID on \$(hostname) at \$(date)\" >&2
export HF_HOME=${HF_CACHE}
export HF_TOKEN=\$(cat ${HF_CACHE}/token 2>/dev/null)
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate ${CONDA_ENV}
python ${HOME_DIR}/${T5} fill \\
    --rays_json ${RAYS_JSON} \\
    --out_dir ${OUT_DIR} \\
    --min_box_samples ${MIN} --adaptive_floor_step ${FLOOR_STEP} \\
    --floor_max ${FLOOR_MAX} \\
    ${COMMON}")
        echo "  fill link ${i}: ${NAME} -> job ${JOBID}"
    done
}

echo "Submitting fill chain: t5_llmint8_accuracy (afterany run job)"
submit_fill_chain t5_sst2_accuracy \
    ${ACC_DIR}/${MODEL_SLUG}__sst2__accuracy__llmint8__cuts3_rays.json \
    ${ACC_DIR}

echo "Submitting fill chain: t5_llmint8_ppl_score (afterany run job)"
submit_fill_chain t5_sst2_ppl_score \
    ${PPL_DIR}/${MODEL_SLUG}__sst2__ppl_score__llmint8__cuts3_rays.json \
    ${PPL_DIR}

echo "Done. Monitor: squeue -u \$USER ; tail ${REPO_ROOT}/logs/t5llmint8_run_*.out logs/fillt5i8_*.out"
