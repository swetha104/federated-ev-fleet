#!/usr/bin/env bash
# Every experiment behind the v2 README. Finished runs (results.json present) are skipped.
#   bash scripts/experiments.sh            # all
#   bash scripts/experiments.sh dropout    # one group: main | dropout | dp | cs
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
GROUP=${1:-all}

run() {  # run <name> <args...>
  local name=$1; shift
  if [[ -f outputs/$name/results.json || -f outputs/$name/baselines.json ]]; then echo "skip $name"; return; fi
  echo "run  $name"
  mkdir -p outputs
  "$PY" -m "$@" --out-dir "outputs/$name" > "outputs/$name.log" 2>&1
}

EV="--kind ev --num-clients 50 --rounds 40 --fraction-train 0.5"

if [[ $GROUP == all || $GROUP == main ]]; then
  run ev50-baselines      evfl.baselines --kind ev --num-clients 50 --epochs 40
  run ev50-fedavg         evfl.run $EV --local-epochs 2
  run ev50-fedadam        evfl.run $EV --local-epochs 2 --strategy fedadam --server-lr 0.01
  run ev50-fedavg-e5-ft   evfl.run $EV --local-epochs 5 --finetune-epochs 5
fi

if [[ $GROUP == all || $GROUP == dropout ]]; then
  for d in 0.3 0.6; do
    run ev50-dropout$d-fedavg  evfl.run $EV --local-epochs 5 --dropout $d
    run ev50-dropout$d-ftavg   evfl.run $EV --local-epochs 5 --dropout $d --strategy fault-tolerant --min-completion-rate 0.6
  done
fi

if [[ $GROUP == all || $GROUP == dp ]]; then
  DP="--kind ev --num-clients 50 --rounds 20 --fraction-train 0.5 --local-epochs 2 --dp-clip 1.0 --dp-delta 1e-4"
  for s in 0.0 0.5 1.0 2.0; do
    run ev50-dp$s  evfl.run $DP --dp-noise $s
  done
fi

if [[ $GROUP == all || $GROUP == cs ]]; then
  CS="--kind cs --data-dir data/cs --num-clients 50 --rounds 40 --fraction-train 0.5 --local-epochs 5"
  run cs-baselines       evfl.baselines --kind cs --data-dir data/cs --num-clients 50 --epochs 40
  run cs-fedavg-ft       evfl.run $CS --finetune-epochs 5
  run cs-dropout0.3-ftavg evfl.run $CS --dropout 0.3 --strategy fault-tolerant --min-completion-rate 0.6
fi
