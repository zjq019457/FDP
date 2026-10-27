import glob
import argparse
import transformers
import torch
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def load_context(fpath="data/PaulGrahamEssays/*.txt", ctx_len=100000):
    context = ""
    for file in glob.glob(fpath):
        with open(file, 'r') as f: 
            context += f.read()
    LLAMA_CHAR_TO_TOKEN_RATIO = 3.66
    context = context[: int(ctx_len * LLAMA_CHAR_TO_TOKEN_RATIO)]
    return context

def insert_needle(context, needle, depth):
    context = context.split(".")
    c_len = len(context)
    needle_place = int(depth * c_len)
    context = ".".join(context[:needle_place]) + "." + needle + ".".join(context[needle_place:])
    return context


def to_device(
    model,
    device,
    enable_tp=False,
    enable_pp=False,
    reverse_device_map=True,
    even_split_layers=True,
):
    if enable_tp and isinstance(device, list):
        if len(device) == 1:
            return model.to(f"cuda:{device[0]}")
        device_ids = [f"cuda:{idx}" for idx in device]
        world_size = len(device_ids)
        tensor_parallel_config = find_predefined_tensor_parallel_config(
            model.config, device_ids
        )
        if tensor_parallel_config is None:
            if model.config.model_type == "mistral":
                tensor_parallel_config = get_mistral_config(model.config, device_ids)
            else:
                tensor_parallel_config = get_default_config(model, device_ids)
        tensor_parallel_config.state_rules[re.compile(r".*full_attention_heads$")] = (
            Split(world_size=world_size, dim=0)
        )
        return tp.tensor_parallel(
            model,
            device_ids,
            tensor_parallel_config=tensor_parallel_config,
            sharded=True,
        )
    elif enable_pp and isinstance(device, list):
        no_split_module_classes = [
            "MistralDecoderLayer",
            "LlamaDecoderLayer",
        ]
        max_memory = {
            device_id: torch.cuda.get_device_properties(device_id).total_memory
            for device_id in device
        }
        print("Max Memory:", max_memory)
        max_memory = get_balanced_memory(
            model,
            max_memory,
            no_split_module_classes=no_split_module_classes,
        )
        device_map = infer_auto_device_map(
            model, max_memory, no_split_module_classes=no_split_module_classes
        )
        modules = list(device_map.keys())
        num_devices = len(device)
        device_map = {}
        current_device_idx = 0

        if even_split_layers:
            num_layers_per_device = model.config.num_hidden_layers / num_devices
            current_layer_idx = 0
            current_other_idx = 0
            for idx, module in enumerate(modules):
                if "layer" in module:
                    device_map[module] = device[current_device_idx]
                    current_layer_idx += 1
                    if current_layer_idx >= num_layers_per_device:
                        current_device_idx += 1
                        current_layer_idx = 0
                elif "lm_head" in module:
                    device_map[module] = device[-1]
                elif "norm" in module:
                    device_map[module] = device[-1]
                elif "embed" in module:
                    device_map[module] = device[0]
                else:
                    device_map[module] = device[current_other_idx]
                    current_other_idx += 1
                    continue
        else:
            num_modules_per_device = len(modules) / num_devices
            for idx, module in enumerate(modules):
                device_map[module] = device[current_device_idx]
                if (idx + 1) >= num_modules_per_device * (current_device_idx + 1):
                    current_device_idx += 1

        if reverse_device_map:
            device_map = {k: num_devices - v - 1 for k, v in device_map.items()}
        print("Device Map:", device_map)
        dispatch_model(model, device_map)
        return model
    else:
        return model.to(device)
