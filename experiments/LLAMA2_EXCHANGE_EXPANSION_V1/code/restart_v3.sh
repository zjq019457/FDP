#!/bin/bash
R=/diskF/jiaqizhang/LLAMA2_EXCHANGE_EXPANSION_V1; cd $R/code
mv $R/quality/rand_e068_20261110 $R/quality/rand_e068_20261110_attempt1_resource
mv $R/logs/q_rand_e068_20261110.log $R/logs/q_rand_e068_20261110.log.attempt1_resource
echo "$(date -Is) q_rand_e068_20261110: refused by free-VRAM safety gate (no evaluation started); archived; drivers v3 (busy-skip fix, resource retry) restarted" >> $R/logs/ENGINEERING_EVENTS.log
(echo "# $(date -Is) drivers v3"; sha256sum l2x_driver.py l2x_driver_floor.py) >> ../PREREG_SHA256SUMS.txt
nohup /home/bai-project/anaconda3/envs/eigen/bin/python -u l2x_driver.py >> ../logs/driver.log 2>&1 < /dev/null &
nohup /home/bai-project/anaconda3/envs/eigen/bin/python -u l2x_driver_floor.py >> ../logs/driver_floor.log 2>&1 < /dev/null &
disown -a
sleep 60
python3 - <<PY
import json
for f in ("STATUS.json","STATUS_FLOOR.json"):
    s=json.load(open("$R/"+f)); print(f,s["at"],s["done"],s["total"],s["running"],s["failed"])
PY
pgrep -af "l2x_quality|l2x_exchange|run_profiles" | grep -v pgrep | awk '{print $4, $5, $6}'
