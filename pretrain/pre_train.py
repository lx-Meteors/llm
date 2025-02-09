from dataclasses import field
from datetime import time

import numpy as np
import pandas as pd
from transformers import DataCollator, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import sys
sys.path.append('/mnt/zhaorunsong/lx/My-LLM')
from datasets import Dataset, load_dataset
from qwen.configuration_qwen import QWenConfig
from qwen.modeling_qwen import QWenLMHeadModel
from qwen.tokenization_qwen import QWenTokenizer

TRAIN_FILES = ["/mnt/zhaorunsong/lx/My-LLM/datasets/wiki.parquet"]
EVAL_FILE = ""

class PreTrainArguments:
    tokenizer_dir: str="../qwen/"
    config_dir: str="../qwen/"
    model_save_dir: str="../model_save/pre/"
    logs_dir: str="./logs/"
    train_files: str = TRAIN_FILES
    eval_file: str = EVAL_FILE
    max_seq_len: int = 64

# 加载预训练参数类
pretrain_args = PreTrainArguments()

tokenizer = QWenTokenizer.from_pretrained(pretrain_args.tokenizer_dir)
tokenizer.pad_token_id = tokenizer.im_end_id

def token_to_id(samples: dict) -> dict:
    batch_text = samples["text"]
    outputs = tokenizer(
        batch_text,
        padding=False,
        return_attention_mask=False,
        truncation=True,
        max_length=pretrain_args.max_seq_len
    )
    input_ids = [np.array(item) for item in outputs["input_ids"]]
    return {"input_ids": input_ids}

def get_map_dataset(files) -> Dataset:
    dataset = load_dataset(
        path="parquet",
        data_files=files,
        split="train",
        cache_dir=".cache",
        keep_in_memory=False
    )
    map_dataset = dataset.map(
        token_to_id,
        batched=True,
        batch_size=10000,
        remove_columns=dataset.column_names,
        num_proc=8,
        keep_in_memory=True
    )
    return map_dataset

# 加载数据集
train_dataset = get_map_dataset(pretrain_args.train_files)
eval_dataset = get_map_dataset(pretrain_args.train_files)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

config = QWenConfig.from_pretrained(pretrain_args.config_dir)
print(config)
model = QWenLMHeadModel(config)

# 统计模型大小
model_size = sum(t.numel() for t in model.parameters())
print(f"QWen size: {model_size / 1000**2:.1f}M parameters")

# 训练
args = TrainingArguments(
    output_dir=pretrain_args.model_save_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=10,
    num_train_epochs=1,
    weight_decay=0.1,
    ddp_find_unused_parameters=False,
    warmup_steps=0,
    learning_rate=1e-4,
    evaluation_strategy="steps",
    eval_steps=1000000000,
    save_steps=50,
    save_strategy="steps",
    save_total_limit=4,
    report_to=None,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=20,
    log_level="info",
    logging_first_step=True,
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")


loss_log = pd.DataFrame(trainer.state.log_history)
loss_log.to_csv(f"./logs/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")


trainer.save_model(pretrain_args.model_save_dir)











