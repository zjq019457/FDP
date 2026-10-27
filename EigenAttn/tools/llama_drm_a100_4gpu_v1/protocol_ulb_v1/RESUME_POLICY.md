# Resume Policy

Each method/task has an atomic state file and an append-safe temporary JSONL. A resume validates protocol ID, model/tokenizer fingerprints, method hash, task manifest, ordered sample IDs, and the last complete JSON line before continuing. Completed sample IDs are skipped and never overwritten. Duplicate or out-of-order IDs stop the affected task.

Task completion performs schema validation, exact expected-count validation, duplicate/missing checks, scoring, fsync, and atomic rename from `.tmp` to canonical output. Detached launches must write `LAUNCH_COMMAND.txt`, `RESUME_COMMAND.txt`, `PID.json`, `RUN_STATE.json`, `PROGRESS.json`, wrapper logs, method logs, and task states. Initialization failures stop the current method; isolated per-sample OOM/failures continue under the status policy.
