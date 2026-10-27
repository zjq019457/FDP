# Multi-GPU Scheduler Audit

- generated_at: `2026-07-28T08:37:07.249698-07:00`
- n_gpus: `4`
- policy: UUID-first exclusive locks; one formal job per GPU; never kill foreign processes.
- recheck_secs: 300
- min_free_mib_default: `70000`

## Inventory

- index=0 uuid=`GPU-2a3f51fa-009a-ce09-ea2e-3a92710b9a07` name=`NVIDIA A100 80GB PCIe` free=81153MiB total=81920MiB
- index=1 uuid=`GPU-f8ce0fbb-2938-371f-4676-4c2c44eaa806` name=`NVIDIA A100 80GB PCIe` free=81153MiB total=81920MiB
- index=2 uuid=`GPU-48dea052-92d1-8022-92eb-09172e488520` name=`NVIDIA A100 80GB PCIe` free=81153MiB total=81920MiB
- index=3 uuid=`GPU-90f68475-4077-7c12-4d66-1cc9092e85c5` name=`NVIDIA A100 80GB PCIe` free=81153MiB total=81920MiB

## Compute apps

