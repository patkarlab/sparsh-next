# Shared job setup, sourced by every PBS script after jobs/settings.sh.
# Loads the CUDA module and activates the conda environment as train_focal_fast.pbs does,
# defines require_file, and sets the determinism variables.

require_file() {  # usage: require_file NAME PATH [HINT]
    if [ ! -f "$2" ]; then
        echo "ERROR: $1 file not found: $2"
        echo "${3:-Fix $1 in jobs/settings.sh, then check with: bash jobs/check_settings.sh}"
        exit 1
    fi
}

if [ -n "${CUDA_MODULE:-}" ]; then
    if command -v module >/dev/null 2>&1; then
        module load "$CUDA_MODULE" || echo "Warning: 'module load $CUDA_MODULE' failed; continuing"
    else
        echo "Warning: no module command in this job; $CUDA_MODULE not loaded"
    fi
fi

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
set +e
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"
activation_status=$?
set -e
if [ "$activation_status" -ne 0 ] || [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
    echo "ERROR: could not activate the conda environment $CONDA_ENV (conda in $CONDA_BASE)."
    echo "Create it as in docs/SETUP.md step 4, or change CONDA_ENV in jobs/settings.sh."
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$RUNS_DIR"
echo "Host $(hostname) | env $CONDA_ENV | python $(command -v python) | $(date)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi
