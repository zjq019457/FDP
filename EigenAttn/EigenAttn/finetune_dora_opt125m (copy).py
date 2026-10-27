import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os

# 创建目录用于保存 Q/K/V 输出
os.makedirs("qkv_outputs", exist_ok=True)

# 全局字典用于保存 Q/K/V
qkv_outputs = {}

# 状态字典，用于在 hook 中访问当前训练步数
trainer_state = {"global_step": 0, "max_steps": 999999}

# 1. 加载模型和分词器
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # OPT 没有 pad_token，设置为 eos_token

# 2. 配置 DoRA / LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 3. 注册 forward hook，只在最后一步保存 Q/K/V
'''
def hook_fn(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if trainer_state["global_step"] == trainer_state["max_steps"]:
                if name not in qkv_outputs:
                    qkv_outputs[name] = []
                qkv_outputs[name].append(output.detach().float().cpu())
    return hook
    '''
def hook_fn(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if trainer_state["global_step"] >= trainer_state["max_steps"] - 1:
                if name not in qkv_outputs:
                    qkv_outputs[name] = []
                qkv_outputs[name].append(output.detach().float().cpu())
    return hook

# 注册 hook 到 q/k/v 层
for name, module in model.named_modules():
    if any(x in name for x in ["q_proj", "k_proj", "v_proj"]):
        module.register_forward_hook(hook_fn(name))

# 4. 加载并格式化数据集
dataset = load_dataset("tatsu-lab/alpaca")["train"]

def format_example(example):
    instruction = example["instruction"]
    input_text = example["input"] or ""
    output_text = example["output"]
    return {
        "text": f"### Instruction:\n{instruction}\n{input_text}\n### Response:\n{output_text}"
    }

dataset = dataset.map(format_example)

# 5. 训练参数设置
training_args = TrainingArguments(
    output_dir="./dora_opt125m_finetune",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    learning_rate=2e-4,
    fp16=True,
)

# 6. 自定义 Trainer 更新 global_step 供 hook 使用
class MyTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        trainer_state["global_step"] = self.state.global_step
        trainer_state["max_steps"] = self.state.max_steps
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

# 7. 构造 Collator
collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    response_template="### Response:",
    instruction_template="### Instruction:"
)

# 8. 启动 Trainer
trainer = MyTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=collator,
    #tokenizer=tokenizer, 
)

# 9. 训练过程
trainer.train()

# 10. 保存最后一个 step 的 Q/K/V
if qkv_outputs:
    torch.save(qkv_outputs, "qkv_outputs/final_step.pt")
    qkv_outputs.clear()

# 11. 保存微调模型和 tokenizer
model.save_pretrained("./dora_opt125m_finetune/final_model")
tokenizer.save_pretrained("./dora_opt125m_finetune/final_model")

