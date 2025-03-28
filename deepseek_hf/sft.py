import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, Trainer, \
    DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model


class SFTArguments():
    model_name = "/openbayes/input/input0/DeepSeek-R1-Distill-Qwen-1.5B"
    train_data_path = "/openbayes/home/opentest/data/chat_dataset.json"
    pretrain_lora_path = "/openbayes/home/opentest/deepseek_hf/results/checkpoint-30000/adapter_model.safetensors"


args = SFTArguments()

# ✅ 修正 train_dataset 取 "train" 数据集
train_dataset = load_dataset("json", data_files=args.train_data_path)
print(train_dataset)
# ✅ 修正 tokenizer 使用方式
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def tokenize_to_id(example: dict) -> dict:
    messages = [
        {"role": "user", "content": example["boy"]},
        {"role": "assistant", "content": example["girl"]}
    ]
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False
    )

    # Tokenize完整对话
    input_ids = tokenizer(
        formatted_text,
        truncation=True,
        max_length=1024,
        return_attention_mask=False,
        add_special_tokens=False,
    )["input_ids"]

    # 查找助理回复的起始位置
    assistant_index = formatted_text.find("<｜Assistant｜>")
    prefix_text = formatted_text[:assistant_index]
    seq_len = len(tokenizer(prefix_text, add_special_tokens=False, max_length=1024, truncation=True)["input_ids"])
    return {"input_ids": input_ids, "seq_len":seq_len}


# ✅ 修正 train_dataset.map
train_dataset = train_dataset["train"]
train_dataset = train_dataset.map(tokenize_to_id, remove_columns=["boy","girl"])
train_dataset = train_dataset.select(range(10000))

def data_collator(features):
    len_ids = [len(feature["input_ids"]) for feature in features]
    input_ids = []
    labels_list = []

    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]

        labels = [-100] * seq_len + ids[seq_len:]

        labels_list.append(torch.LongTensor(labels))
        input_ids.append(torch.LongTensor(ids))
        if len(labels) != len(ids):
            print()
    return {"input_ids": torch.stack(input_ids), "labels": torch.stack(labels_list)}

training_args = TrainingArguments(
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
)

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
state_dict = load_file(args.pretrain_lora_path)
deepseek.load_state_dict(state_dict, strict=False)

deepseek.print_trainable_parameters()

trainer = Trainer(
    model=deepseek,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    tokenizer=tokenizer
)

trainer.train()
