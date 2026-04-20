#!/bin/bash
# Train + test-eval every model in the benchmark.
# Writes per-step logs under logs/, checkpoints under checkpoints/<run_id>/,
# and evaluation metrics under results/<run_id>/.
#
# Resumable: skips TRAIN if a checkpoint already exists, skips EVAL if a
# valid metrics.json is already written.
#
# Usage:
#     bash scripts/run_all.sh               # bootstrap=100 (default)
#     BOOTSTRAP=1000 bash scripts/run_all.sh
set -u

mkdir -p logs results

BOOTSTRAP="${BOOTSTRAP:-100}"

# "config_name|run_id" pairs. run_id also names the checkpoints/ and results/
# subdirectories. Edit freely — the script loops over whatever you list here.
RUNS=(
  "lgbm|lgbm"
  "lgbm_no_addr|lgbm_noaddr"
  "lgbm_no_identity|lgbm_nocat"
  "mlp|mlp"
  "mlp_no_identity|mlp_nocat"
  "lstm|lstm"
  "transformer|transformer"
  "mamba3_constant|mamba3_const"
  "mamba3_physical|mamba3_phys"
  "mamba3_physical_no_identity|mamba3_phys_nocat"
)

has_train_artifact() {
    local run_id="$1"
    [ -d "results/${run_id}/model" ] && return 0
    [ -f "checkpoints/${run_id}/last.ckpt" ] && return 0
    compgen -G "checkpoints/${run_id}/epoch_*.ckpt" > /dev/null && return 0
    return 1
}

has_valid_metrics() {
    local run_id="$1"
    local m="results/${run_id}/metrics.json"
    [ -f "${m}" ] || return 1
    python -c "import json,sys; d=json.load(open('${m}')); sys.exit(0 if 'revert_auc' in d else 1)" 2>/dev/null
}

run_one() {
    local cfg="$1" run_id="$2"
    local log="logs/${run_id}.log"

    if has_train_artifact "${run_id}"; then
        echo ">>> [$(date -Iseconds)] SKIP TRAIN ${run_id} (artifact exists)"
    else
        echo ">>> [$(date -Iseconds)] TRAIN ${run_id} (configs/${cfg}.yaml)"
        # Pass --ckpt-dir=checkpoints (NOT checkpoints/${run_id}); train.py
        # appends args.run_id internally, so adding it here would nest paths
        # like checkpoints/${run_id}/${run_id}/last.ckpt that the EVAL step
        # would silently miss, loading a random model instead.
        python -u -m src.training.train \
            --config "configs/${cfg}.yaml" \
            --run-id "${run_id}" \
            --data-dir data/processed \
            --ckpt-dir checkpoints \
            > "${log}.train" 2>&1
        local train_rc=$?
        echo ">>> [$(date -Iseconds)] TRAIN ${run_id} exit=${train_rc}"
        if [ ${train_rc} -ne 0 ]; then
            echo "!!! TRAIN FAILED for ${run_id} — skipping eval"
            return ${train_rc}
        fi
    fi

    if has_valid_metrics "${run_id}"; then
        echo ">>> [$(date -Iseconds)] SKIP EVAL ${run_id} (metrics.json valid)"
        return 0
    fi

    echo ">>> [$(date -Iseconds)] EVAL  ${run_id} (test split, bootstrap ${BOOTSTRAP})"
    local ckpt_arg=""
    local ckpt_found=""
    for candidate in \
        "checkpoints/${run_id}/last.ckpt" \
        "checkpoints/${run_id}/${run_id}/last.ckpt"; do
        if [ -f "${candidate}" ]; then
            ckpt_found="${candidate}"
            break
        fi
    done
    if [ -z "${ckpt_found}" ]; then
        # Fall back to the most-recent epoch_*.ckpt.
        ckpt_found=$(ls -t "checkpoints/${run_id}/"epoch_*.ckpt \
                           "checkpoints/${run_id}/${run_id}/"epoch_*.ckpt \
                           2>/dev/null | head -1)
    fi
    # For sklearn families, no --checkpoint is needed; run_eval.py loads the
    # pickle from results/<run_id>/model/. Detect family from the YAML.
    local family
    family=$(python -c "import yaml; print(yaml.safe_load(open('configs/${cfg}.yaml'))['model']['family'])" 2>/dev/null)
    if [ -n "${ckpt_found}" ]; then
        ckpt_arg="--checkpoint ${ckpt_found}"
    elif [ "${family}" = "neural" ]; then
        # Abort this EVAL rather than silently evaluate a random-init model.
        echo "!!! EVAL ABORTED for ${run_id}: no checkpoint found under checkpoints/${run_id}/" >&2
        return 2
    fi
    python -u -m src.evaluation.run_eval \
        --config "configs/${cfg}.yaml" \
        --run-id "${run_id}" \
        --split test \
        --bootstrap "${BOOTSTRAP}" \
        ${ckpt_arg} \
        > "${log}.eval" 2>&1
    local eval_rc=$?
    echo ">>> [$(date -Iseconds)] EVAL  ${run_id} exit=${eval_rc}"
    return ${eval_rc}
}

echo "=== Benchmark run started: $(date -Iseconds) (bootstrap=${BOOTSTRAP}) ==="
for spec in "${RUNS[@]}"; do
    cfg="${spec%|*}"
    run_id="${spec#*|}"
    run_one "${cfg}" "${run_id}" || echo "(continuing despite failure)"
done
echo "=== Benchmark run finished: $(date -Iseconds) ==="
