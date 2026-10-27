import torch
import torch.nn as nn
from typing import Dict, List, Optional
from functools import partial
import subprocess
import re
import os
import time
import pdb


def nvidia_smi_memory_info():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    output = result.stdout.split("\n")[:-1]

    gpu_memory_info = []
    for line in output:
        gpu_id, total_memory, used_memory, free_memory = map(int, re.split(",\s", line))
        gpu_memory_info.append(
            {
                "id": gpu_id,
                "total_memory": total_memory,
                "used_memory": used_memory,
                "free_memory": free_memory,
            }
        )

    return gpu_memory_info


num_gpus = torch.cuda.device_count()


def get_gpu_memory():
    memory_info = []
    gpu_memory_info = nvidia_smi_memory_info()

    try:
        gpu_index = [int(k) for k in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    except KeyError:
        gpu_index = [x["id"] for x in gpu_memory_info]

    for gpu_id, i in enumerate( gpu_index):
        gpu = gpu_memory_info[i]
        total_memory = gpu["total_memory"]
        
        used_memory = gpu["used_memory"]
        memory_info.append((gpu_id, total_memory, used_memory))
    return memory_info


def get_lowest_occupied_gpu(wait_memory=1000):

    now_lowest_memory = 1e9
    while now_lowest_memory > wait_memory:
        if not now_lowest_memory == 1e9:
            time.sleep(10)
        memory_info = get_gpu_memory()
        gpu_id, tot_mem, used_mem = sorted(
            memory_info, key=lambda x: x[2], reverse=False
        )[0]
        now_lowest_memory = used_mem

    return gpu_id


def sort_layers_by_params(layers: List[nn.Module]):
    return sorted(
        layers, key=lambda m: sum(p.numel() for p in m.parameters()), reverse=True
    )


def get_all_gpu_free_memory():
    return sum(
        [
            total_memory - used_memory
            for gpu_id, total_memory, used_memory in get_gpu_memory()
        ]
    )


def assign_layers_to_gpus(
    layers: List[nn.Module],
    reserve_mb: int = 2048,
    logger: Optional[object] = None,
):
    layer_gpu_map = {}
    weight_num = 0
    for module in layers:
        if hasattr(module, "weight"):
            weight_num += module.weight.numel()
    weight_mb = weight_num * 2 / 1024 / 1024
    all_gpu_mems = get_all_gpu_free_memory()
    while all_gpu_mems < weight_mb * 1.3:
        time.sleep(10)
        all_gpu_mems = get_all_gpu_free_memory()

    gpu_order = [gpu_id for gpu_id, _, _ in get_gpu_memory()]
    estimated_usage_mb: Dict[int, float] = {gpu_id: 0.0 for gpu_id in gpu_order}
    first_gpu_id = None

    for i, layer in enumerate(layers):
        if i == len(layers) - 1 and first_gpu_id is not None:
            layer_gpu_map[layer] = first_gpu_id
            layer.to(f"cuda:{first_gpu_id}")
            layer.device = f"cuda:{first_gpu_id}"
            msg = f"map last layer {i} to gpu {first_gpu_id} (wrap back for shared input/output device)"
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)
            continue

        layer_memory = sum(p.element_size() * p.numel() for p in layer.parameters()) / 1024**2
        available_gpus = get_gpu_memory()
        gpu_stats = {gpu_id: (tot_memory, allocated_memory) for gpu_id, tot_memory, allocated_memory in available_gpus}

        candidates = []
        for gpu_id in gpu_order:
            tot_memory, allocated_memory = gpu_stats[gpu_id]
            free_memory = tot_memory - allocated_memory
            effective_free = free_memory - reserve_mb
            if effective_free <= layer_memory:
                continue
            projected_pressure = (allocated_memory + estimated_usage_mb[gpu_id] + layer_memory) / max(tot_memory, 1)
            candidates.append((projected_pressure, estimated_usage_mb[gpu_id], gpu_id, free_memory))

        if not candidates:
            raise RuntimeError(
                f"memory not enough for layer {i}: need~{layer_memory:.1f}MB with reserve {reserve_mb}MB, "
                f"available={available_gpus}"
            )

        _, _, chosen_gpu, free_memory = min(candidates)
        layer_gpu_map[layer] = chosen_gpu
        layer.to(f"cuda:{chosen_gpu}")
        layer.device = f"cuda:{chosen_gpu}"
        estimated_usage_mb[chosen_gpu] += layer_memory
        if first_gpu_id is None:
            first_gpu_id = chosen_gpu

        msg = (
            f"map layer {i} to gpu {chosen_gpu}, layer_mb={layer_memory:.1f}, "
            f"free_mb={free_memory:.1f}, est_gpu_load_mb={estimated_usage_mb[chosen_gpu]:.1f}"
        )
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    return layer_gpu_map


# forward hook
def forward_hook_wrapper(gpu_id):
    def forward_hook(module, input, kwargs):
        # breakpoint()
        input = tuple(_.to(f"cuda:{gpu_id}") for _ in input)
        kwargs = {
            k: v.to(f"cuda:{gpu_id}") if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        return input, kwargs

    return forward_hook


def add_forward_hooks(layer_gpu_map):
    prev_gpu_id = None
    for layer, gpu_id in layer_gpu_map.items():
        layer: nn.Module
        if prev_gpu_id is None:
            prev_gpu_id = gpu_id
        # if gpu_id != prev_gpu_id:
        layer.register_forward_pre_hook(forward_hook_wrapper(gpu_id), with_kwargs=True)
        prev_gpu_id = gpu_id


def map_layers_to_multi_gpus(layers, logger: Optional[object] = None):
    layer_gpu_map = assign_layers_to_gpus(layers, logger=logger)
    add_forward_hooks(layer_gpu_map)
    return layer_gpu_map


if __name__ == "__main__":
    info = get_gpu_memory()
    print(info)
