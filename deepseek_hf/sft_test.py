import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, Trainer
from peft import LoraConfig, get_peft_model
class SFTArguments():
    model_name = "/openbayes/input/input0/DeepSeek-R1-Distill-Qwen-1.5B"
    sft_lora_path = "/openbayes/home/opentest/deepseek_hf/sft_test_results/checkpoint-30000/adapter_model.safetensors"

args = SFTArguments()
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
deepseek = AutoModelForCausalLM.from_pretrained(args.model_name)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
deepseek = get_peft_model(deepseek, lora_config)
# ✅ 修正 load_state_dict
state_dict = load_file(args.sft_lora_path)
deepseek.load_state_dict(state_dict, strict=False)

text = "<｜User｜>人要是上进 善良就好"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
device = "cuda"  # 或 "cpu" 根据你的环境
deepseek.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = deepseek.generate(**inputs, max_length=256, do_sample=True, temperature=0.7, top_p=0.9)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

