#!/usr/bin/env bash
OUT="$1"; INT="${2:-5}"
echo "time,container,cpu_perc,mem_usage,mem_perc,net_io,block_io,pids" > "$OUT"
while true; do
  docker stats --no-stream --format \
    '{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}},{{.PIDs}}' 2>/dev/null \
    | grep -E "server|client|prep" | sed "s/^/$(date +%s),/" >> "$OUT"
  sleep "$INT"
done
