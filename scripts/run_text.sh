#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MIMIC="${1:-./data/mimic3}"; KGDIR="${2:-./GraphCare-main/KG_mapping}"
export KG_RESOURCES="${KG_RESOURCES:-$KGDIR/../resources}"
SECTION="${SECTION:-diagnosis}"; T="notes_$SECTION"; W="work/$T"; R="results/text_$T"; PY="python3 fedrag_sim.py"
mkdir -p "$W" "$R"
mkdir -p work
[ -f work/measurements.pkl ] || $PY measurements --mimic "$MIMIC" --out work/measurements.pkl
[ -f "$W/cohort.pkl" ] || $PY notes --mimic "$MIMIC" --section "$SECTION" --measurements work/measurements.pkl --out "$W/cohort.pkl"
[ -f "$W/kg.pkl" ]     || $PY kg --cohort "$W/cohort.pkl" --kgdir "$KGDIR" --out "$W/kg.pkl"
$PY sweep_text --cohort "$W/cohort.pkl" --kg "$W/kg.pkl" --kgdir "$KGDIR" --out "$R" --rounds "${ROUNDS:-5}" \
    --task_tag "$T" --threads "${THREADS:-4}" --clients ${CLIENT_COUNTS:-10} --seeds ${SEEDS:-1 2 3} \
    --hide_dx ${HIDE_DX:-0 1} --extra "--tg_model ${TG_MODEL:-google/flan-t5-small} --tg_max_train ${MAX_TRAIN:-400} --tg_max_test ${MAX_TEST:-100}"
$PY tables --results "$R" --out "$R/tables.tex"
echo "Tables: $R/tables.tex"
