#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TASK="${TASK:-mortality_next}" ARM="${ARM:-client}" FL="${FL:-fedavg}" MODEL="${MODEL:-sage}"
export CLIENTS="${CLIENTS:-10}" SEED="${SEED:-1}" PARTITION="${PARTITION:-noniid}" HYPOTHESES="${HYPOTHESES:-1}"
RES="${RESULTS_HOST:-./results}"; mkdir -p "$RES" "${WORK_HOST:-./work}"; chmod 777 "$RES" "${WORK_HOST:-./work}" 2>/dev/null || true
export FQP_HOPS="${FQP_HOPS:-2}" FQP_DECOYS="${FQP_DECOYS:-0}" FQP_REPL="${FQP_REPL:-2}" MEAS="${MEAS:-1}" HIDE_DX="${HIDE_DX:-0}"
SFX=""; [ "$ARM" = fqp ] && [ "$FQP_HOPS" != 2 ] && SFX="${SFX}_h$FQP_HOPS"
[ "$ARM" = fqp ] && [ "$FQP_DECOYS" != 0 ] && SFX="${SFX}_d$FQP_DECOYS"
{ [ "$ARM" = fqp ] || [ "$ARM" = client_local ]; } && [ "$FQP_REPL" != 2 ] && SFX="${SFX}_r$FQP_REPL"
case "$TASK" in notes*) [ "$HIDE_DX" != 0 ] && SFX="${SFX}_hidedx";; esac
[ "$MEAS" = 0 ] && SFX="${SFX}_nomeas"
case "$TASK" in notes*) MTAG=t5lora;; *) MTAG=$MODEL;; esac
TAG="${TASK}__${FL}__${MTAG}__${ARM}${SFX}__c${CLIENTS}__${PARTITION}__seed${SEED}"
if [ -f "$RES/$TAG.summary.json" ] && [ -z "${FORCE:-}" ]; then echo "skip (done) $TAG"; exit 0; fi
P="fedrag_$$"
docker compose -p "$P" build -q
docker compose -p "$P" run --rm prep
bash scripts/docker_stats.sh "$RES/$TAG.docker_stats.csv" 5 & STATS=$!
START=$(date +%s)
docker compose -p "$P" up -d --no-deps server
docker compose -p "$P" up -d --no-deps --scale client="$CLIENTS" client
docker wait "$(docker compose -p "$P" ps -q server)" >/dev/null
END=$(date +%s)
docker compose -p "$P" logs --no-color > "$RES/$TAG.containers.log" 2>&1 || true
kill $STATS 2>/dev/null || true
docker compose -p "$P" down -v --remove-orphans >/dev/null
echo "{\"tag\": \"$TAG\", \"wall_s\": $((END-START))}" > "$RES/$TAG.docker_wall.json"
echo "finished $TAG in $((END-START)) s"
