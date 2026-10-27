# Historical Command Reconstruction

- Current reconstruction basis: source code flags, builder defaults, local artifact names and legacy logs.
- This file is intentionally conservative and will not invent missing submission-time flags.

## Immediate command clues

- `step1_diagnose_llama.py --base <dense> --cmp <final_model> --tok <tokenizer> --out <json>`
- `per_layer_eb_from_step1.py --step1 <json> --out <json> --eb_base <...> --lambda_eb <...> [--normalize mean]`
- `build_eigen_basis_llama_error_budget_offline.py --model <dense> --data_dir <wikitext> --out_dir <dir> [--per_layer_eb_json <json>]`
- `sanity_ppl_dual_protocol.py --cmp <final_model> --tok <Llama-3> --arch llama --dtype bf16`
