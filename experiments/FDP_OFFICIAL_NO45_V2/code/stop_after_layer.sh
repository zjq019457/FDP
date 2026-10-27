#!/usr/bin/env bash
# Send SIGINT (atexit dumps the trace) to our fulltrace_no45 run once layer 23 has finished; layers 24-31 come from the parallel parts.
N=/diskF/jiaqizhang/FDP_OFFICIAL_NO45_V2; LOG=$N/runs/fulltrace_no45/stdout.log
until grep -q "^20.*layer 23 error" $LOG; do sleep 20; done
PID=$(ps -C python -o pid=,args= | awk '/run_shim_v2.py/ && /fulltrace_no45/ {print $1}')
echo "$(date -Is) layer 23 done; SIGINT -> pid=$PID" >> $N/logs/driver.log
[ -n "$PID" ] && kill -INT $PID
