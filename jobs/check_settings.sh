#!/bin/bash
# Check jobs/settings.sh on the login node before submitting jobs. From the sparsh-next folder:
#   bash jobs/check_settings.sh
# Only reads files, apart from creating RUNS_DIR if it does not exist yet.
cd "$(dirname "$0")/.." || exit 1
source "${SETTINGS_FILE:-jobs/settings.sh}"
problems=0

report_file() {  # usage: report_file NAME PATH REQUIRED(yes|no)
    if [ -z "$2" ]; then
        if [ "$3" = yes ]; then
            echo "PROBLEM  $1 is empty"; problems=$((problems + 1))
        else
            echo "ok       $1 not set (optional)"
        fi
    elif [ -f "$2" ]; then
        echo "ok       $1 = $2 ($(du -h "$2" | cut -f1))"
    else
        echo "PROBLEM  $1 = $2 does not exist"; problems=$((problems + 1))
    fi
}

report_file DATA_PATH "$DATA_PATH" yes
report_file EXCLUDE_IDS "$EXCLUDE_IDS" no
if [ -n "$EXCLUDE_IDS" ] && [ -f "$EXCLUDE_IDS" ]; then
    echo "         $(grep -cv '^[[:space:]]*$' "$EXCLUDE_IDS") non-empty lines; the first three:"
    head -n 3 "$EXCLUDE_IDS" | sed 's/^/           /'
fi
report_file GROUPS_FILE "$GROUPS_FILE" no

if [ -d "$CONDA_BASE/envs/$CONDA_ENV" ]; then
    echo "ok       conda environment $CONDA_ENV"
else
    echo "PROBLEM  conda environment $CONDA_ENV not found in $CONDA_BASE/envs (docs/SETUP.md step 4)"
    problems=$((problems + 1))
fi

if mkdir -p "$RUNS_DIR" 2>/dev/null && [ -w "$RUNS_DIR" ]; then
    echo "ok       RUNS_DIR = $RUNS_DIR ($(df -Ph "$RUNS_DIR" | awk 'NR == 2 {print $4}') free on that file system)"
    if command -v lfs >/dev/null 2>&1; then
        mount_point=$(df -P "$RUNS_DIR" | awk 'NR == 2 {print $6}')
        echo "         Your quota there (limit 0 or - means no limit):"
        lfs quota -h -u "${USER:-$(id -un)}" "$mount_point" 2>/dev/null | sed 's/^/           /'
    fi
else
    echo "PROBLEM  RUNS_DIR = $RUNS_DIR is not writable"; problems=$((problems + 1))
fi
echo "         Classes left out: ${EXCLUDE_PREFIXES:-none}"
echo "         Training queues, in turn: ${TRAIN_QUEUES:-a40}"

if [ "$problems" -eq 0 ]; then
    echo "SETTINGS OK"
else
    echo "$problems problem(s): fix them in jobs/settings.sh and run this check again"
    exit 1
fi
