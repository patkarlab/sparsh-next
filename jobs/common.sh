# Shared job setup: activate the SPARSH-next environment and set determinism variables.
# If activation fails on your cluster, replace the block below with the activation
# lines from a PBS script that already works for you, keeping CONDA_ENV.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
fi
conda activate "$CONDA_ENV"
module load cuda/12.3 2>/dev/null || true

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$RUNS_DIR"
echo "Host $(hostname) | env $CONDA_ENV | python $(which python) | $(date)"
