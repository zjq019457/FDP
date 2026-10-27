# PALU_FISHER_VS_FUNCTIONAL_MARGINAL_V1

Status: **PALU_FISHER_VS_FUNCTIONAL_MARGINAL_COMPLETE**

This independent branch does not modify historical frozen results, q, fragility, resource locks, or the PaLu native Fisher implementation.

## Frozen Llama-3 / current Fixed audit

| tier | Fisher vs phi Spearman | q vs phi Spearman | mean phi receiver | mean phi donor | exact rank |
|---|---:|---:|---:|---:|---|
| LIGHT | -0.189883 | -0.0909091 | 0.00932242 | 0.0102122 | True |
| MEDIUM | 0.0414223 | 0.507331 | 0.0647542 | 0.0276757 | True |
| AGGRESSIVE | 0.611804 | 0.497067 | 0.247496 | 0.0586554 | True |

The frozen roots have aggregate schedules but no immutable donor→receiver event-pair manifest, so exact per-event transfer-consistency is not claimed.

## Llama-2 official schedule freeze

| ratio | integer target | realized native rank | schedule SHA |
|---:|---:|---:|---|
| 0.7 | 45875 | 45760 | `7c460aa5e48452bb98e474530993a3239042f38d2e12e991309b7fc16e1f89b4` |
| 0.5 | 32768 | 32640 | `9893cc06dbfae4442b11e0e5f5620d8464c6e23921af24f85422a230d00d78a4` |
| 0.3 | 19661 | 19584 | `31ffd70e8c64fcc08ac6fe980b18ea3bfb4af411f31ebdc44b73eb299060c10e` |

0.7 reproduces the immutable anchor exactly: **True**.

The checked-out wrapper path is quarantined because it exposes 8 slots/module and 262144 capacity, versus the immutable anchor schema of 2 slots/module and 65536 capacity.

## Llama-2 marginal profile

| ratio | status | baseline J | completed units | legal-min units |
|---:|---|---:|---:|---:|
| 0.7 | MARGINAL_PROFILE_COMPLETE | 2.3842269234035323 | 128 | 0 |
| 0.5 | MARGINAL_PROFILE_COMPLETE | 2.649117399438393 | 128 | 0 |
| 0.3 | MARGINAL_PROFILE_COMPLETE | 3.4732968507886923 | 86 | 42 |

For formal FDP realization, the transfer-count policy remains `TRANSFER_COUNT_POLICY_NOT_FROZEN`; no formal transfer quality branch is launched from this task.

## Outputs

- `PALU_NATIVE_MARGINAL_PER_UNIT.csv`
- `PALU_SIGNAL_CORRELATION_TABLE.csv`
- `PALU_FISHER_VS_FUNCTIONAL_MARGINAL.json`
- `PALU_FISHER_VS_FUNCTIONAL_MARGINAL.md`
