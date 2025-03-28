import sys
from dataclasses import dataclass
from typing import Dict

from datasets import load_dataset, Dataset
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig,DPOTrainer

def get_dataset(file) -> Dataset:
    dataset = load_dataset('json', split="train", data_files=file)
    def split_prompt_responses(sample) -> Dict[str, str]:
        return {
            "prompt": f"{sample['boy']}<｜end▁of▁sentence｜>",
            "chosen": f"{sample['boy']}<｜end▁of▁sentence｜>",
            "rejected": f"{sample['girl']}<｜end▁of▁sentence｜>"
        }
    return dataset.map(function=split_prompt_responses).shuffle(12345)

def train_dpo(config, peft_config):
    model_name = "/openbayes/input/input0/DeepSeek-R1-Distill-Qwen-1.5B"
    sft_lora_path = "/openbayes/home/opentest/deepseek_hf/sft_test_results/checkpoint-30000/adapter_model.safetensors"
    # step 1. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # step 2. 加载SFT模型
    model_train = AutoModelForCausalLM.from_pretrained(model_name)
    state_dict = load_file(sft_lora_path)
    model_train.load_state_dict(state_dict, strict=False)

    model_ref = AutoModelForCausalLM.from_pretrained(model_name)
    state_dict = load_file(sft_lora_path)
    model_ref.load_state_dict(state_dict, strict=False)

    train_dataset = get_dataset("/openbayes/home/opentest/data/train_chat_dataset.json")
    eval_dataset = get_dataset( "/openbayes/home/opentest/data/train_chat_dataset.json")

    dpo_trainer = DPOTrainer(
        model_train,
        model_ref,
        peft_config=peft_config,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer
        # data_collator=data_collator
    )

    dpo_trainer.train()
    dpo_trainer.save_model(config.model_save_dir)



if __name__ == '__main__':

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,  # text 2 text lora model
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="all",
    )

    dpo_config = DPOConfig(
        output_dir='./sft_test_results',
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        remove_unused_columns=False,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./sft_logs',
        logging_steps=10,
        evaluation_strategy='steps',
        save_strategy='epoch',
        learning_rate=5e-5,
        eval_steps=1000000,
        force_use_ref_model=True
    )

    train_dpo(dpo_config, peft_config)