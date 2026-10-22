#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MIMIC="${1:-./data/mimic3}"; KGDIR="${2:-./GraphCare-main/KG_mapping}"
export KG_RESOURCES="${KG_RESOURCES:-$KGDIR/../resources}"
TASK="${TASK:-mortality_next}"; WORK="${WORK:-work/$TASK}"; RES="${RES:-results/sim_$TASK}"
USER_DATA="${USER_DATA:-./data/user_data}"; PY="python3 fedrag_sim.py"
mkdir -p "$WORK" "$RES" "$USER_DATA"
[ -f work/measurements.pkl ] || $PY measurements --mimic "$MIMIC" --out work/measurements.pkl
[ -f "$WORK/cohort.pkl" ] || $PY build --mimic "$MIMIC" --task "$TASK" --exclude_file "$USER_DATA/reserved_subjects.txt" --measurements work/measurements.pkl --out "$WORK/cohort.pkl"
[ -f "$WORK/kg.pkl" ]     || $PY kg --cohort "$WORK/cohort.pkl" --kgdir "$KGDIR" --user_data "$USER_DATA" --out "$WORK/kg.pkl"
[ -f "$RES/retrieval.json" ] || $PY retrieval --cohort "$WORK/cohort.pkl" --kg "$WORK/kg.pkl" \
    --clients ${CLIENT_COUNTS:-10} --seeds ${SEEDS:-1 2 3} --out "$RES/retrieval.json"
$PY sweep --cohort "$WORK/cohort.pkl" --kg "$WORK/kg.pkl" --out "$RES" --grid "${GRID:-all}" \
    --clients ${CLIENT_COUNTS:-10} --seeds ${SEEDS:-1 2 3} --rounds "${ROUNDS:-20}" --budget "${BUDGET:-16}" \
    --task_tag "$TASK" --threads "${THREADS:-2}" --comm_dtype "${COMM_DTYPE:-fp32}" \
    --kgc_rounds "${KGC_ROUNDS:-10}" --user_data "$USER_DATA" --hypotheses
$PY tables --results "$RES" --out "$RES/tables.tex"
echo "Tables: $RES/tables.tex   every metric of every run: $RES/all_runs.csv"
