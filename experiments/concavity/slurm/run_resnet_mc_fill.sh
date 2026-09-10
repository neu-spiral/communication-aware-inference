#!/bin/bash
# =============================================================================
# run_resnet_mc_fill.sh  (2026-07-18)
#
# Monte-Carlo random-ray concavity + adaptive fill for the ResNet56/CIFAR-10
# backend, one SLURM job per compressor (topk, quantization, llmint8). Produces
# mc_concavity-format rays.json with >=100 usable rays per eta-floor up to 0.7,
# which jensen_concavity merges with the 10 directional-profile rays under
# outputs/concavity_resnet (same (model,dataset,metric,n_cuts,strategy) key).
#
# The ResNet accuracy eval is cheap (a couple fixed CIFAR batches on resnet56),
# so base + fill run in one job. Backend config (checkpoint, cutpoints 8,14,21)
# matches the original directional-profile runs.
#
# CIFAR-10: expects data/cifar-10-python.tar.gz already present (torchvision
# extracts it); set RESNET_DOWNLOAD=1 to fetch on the compute node otherwise.
#
# Usage: bash experiments/concavity/slurm/run_resnet_mc_fill.sh
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
N_RAYS=120
N_POINTS=20
TIME="04:00:00"
MEM="32G"
mkdir -p "${REPO_ROOT}/logs"

OUT="${REPO_ROOT}/outputs/mc_concavity/resnet/imagenet_accuracy"

submit() {
    local COMP=$1
    local OUT_DIR=${OUT}/${COMP}/cuts3
    local JOBID
    JOBID=$(sbatch --parsable \
        --partition=gpu --gres=gpu:a100:1 \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=${MEM} --time=${TIME} \
        --job-name=resnetmc_${COMP} \
        --output=${REPO_ROOT}/logs/resnetmc_${COMP}_%j.out \
        --error=${REPO_ROOT}/logs/resnetmc_${COMP}_%j.err \
        --wrap="
set -e
echo \"[wrap] resnet MC fill ${COMP} Job \$SLURM_JOB_ID on \$(hostname) at \$(date)\" >&2
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate ${CONDA_ENV}
export COMPRESSOR_BACKEND=gpu
python ${HOME_DIR}/experiments/concavity/resnet_mc_concavity.py \\
    --compressor ${COMP} --compressor_backend gpu \\
    --checkpoint ${HOME_DIR}/assets/resnet56-4bfd9763.th \\
    --data_root ${HOME_DIR}/data --cutpoints 8,14,21 \\
    --fast_batches 1 --n_repeat 1 \\
    --n_rays ${N_RAYS} --n_points ${N_POINTS} \\
    --min_box_samples ${MIN} --adaptive_floor_step ${FLOOR_STEP} \\
    --floor_max ${FLOOR_MAX} --tol 0.005 --seed 0 \\
    --out_dir ${OUT_DIR}")
    echo "  ${COMP} -> job ${JOBID}"
}

for COMP in topk quantization llmint8; do
    submit ${COMP}
done
echo "Done. Monitor: squeue -u \$USER ; tail ${REPO_ROOT}/logs/resnetmc_*.out"
