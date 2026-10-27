# save as: dump_transform_report.py
import json, statistics

def eval_one(layer:int, kind:str):
    """
    你把自己现有的逻辑塞进来，返回一个 dict，至少包含：
    {
      "U_shape":[7168,448],
      "orth_err_raw":...,
      "left":{"weight_rel_fro":..., "act_cos_mean":..., "act_relL2_mean":..., "qr_orth_err":...},
      "right":{...},
      "better_side":"left/right"
    }
    """
    raise NotImplementedError

def main(n_layers=48):
    rep = {}
    for layer in range(n_layers):
        rep[layer] = {}
        for kind in ["q","k","v"]:
            r = eval_one(layer, kind)
            rep[layer][kind] = r

    # 汇总统计
    better = []
    dcos = []
    dfro = []
    for layer in rep:
        for kind in rep[layer]:
            r = rep[layer][kind]
            better.append(r["better_side"])
            dcos.append(r["right"]["act_cos_mean"] - r["left"]["act_cos_mean"])
            dfro.append(r["left"]["weight_rel_fro"] - r["right"]["weight_rel_fro"])  # >0 表示 right 更好(误差更小)

    summary = {
        "better_side_counts": {s: better.count(s) for s in sorted(set(better))},
        "delta_act_cos_mean": {"mean": statistics.mean(dcos), "min": min(dcos), "max": max(dcos)},
        "delta_weight_rel_fro_improve": {"mean": statistics.mean(dfro), "min": min(dfro), "max": max(dfro)},
    }
    out = {"summary": summary, "detail": rep}

    with open("transform_report.json","w") as f:
        json.dump(out, f, indent=2)
    print("Wrote transform_report.json")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
