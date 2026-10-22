#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
for SEED in ${SEEDS:-1 2 3}; do for CLIENTS in ${CLIENT_COUNTS:-10}; do
  for C in fedavg:sage:none fedavg:sage:random fedavg:sage:server fedavg:sage:client \
           fedavg:sage:client_local fedavg:sage:fqp parc:sage:fqp fedprox:sage:fqp \
           fedprox:sage:none fedprox:sage:server fedprox:sage:client fedcurv:sage:none fedcurv:sage:server fedcurv:sage:client \
           fedlwf:sage:none fedlwf:sage:server fedlwf:sage:client parc:sage:none parc:sage:server parc:sage:client \
           fedavg:bag:none fedavg:bag:client fedavg:retain:none fedavg:retain:client fedavg:transformer:none fedavg:transformer:client; do
    IFS=: read FL MODEL ARM <<< "$C"
    HYP=0; [ "$FL$MODEL" = "fedavgsage" ] && { [ "$ARM" = client ] || [ "$ARM" = none ] || [ "$ARM" = fqp ]; } && HYP=1
    FL=$FL MODEL=$MODEL ARM=$ARM CLIENTS=$CLIENTS SEED=$SEED HYPOTHESES=$HYP bash scripts/docker_run.sh || echo "FAILED $C"
  done
  for ARM in none server client fqp; do
    FL=fedavg MODEL=sage ARM=$ARM CLIENTS=$CLIENTS SEED=$SEED HYPOTHESES=0 MEAS=0 bash scripts/docker_run.sh || echo "FAILED nomeas $ARM"
  done; done; done
docker run --rm -v "$(cd "${RESULTS_HOST:-./results}" && pwd)":/results --entrypoint python fedrag:cpu fedrag_sim.py tables --results /results --out /results/tables.tex
