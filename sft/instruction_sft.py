import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import sys
sys.path.append('/mnt/zhaorunsong/lx/My-LLM')
from qwen.modeling_qwen import QWenLMHeadModel
from qwen.tokenization_qwen import QWenTokenizer


class SFTArguments:
    SFT_FILES = "/mnt/zhaorunsong/lx/My-LLM/datasets/alpaca.parquet"
    model_path = "/mnt/zhaorunsong/lx/My-LLM/model_save/pre/checkpoint-650"
    model_save_dir = "/mnt/zhaorunsong/lx/My-LLM/model_save/sft/"
    max_seq_len = 512

sftArguments = SFTArguments()

PROMPT_DICT = {
    "prompt_input": "你是一个助手 " "用户: {instruction} {input} 回答: ",
    "prompt_no_input": "你是一个助手 " "用户: {instruction}  回答: ",
}

dataset = load_dataset(path="parquet", data_files=sftArguments.SFT_FILES, split="train", keep_in_memory=False)
tokenizer = QWenTokenizer.from_pretrained(sftArguments.model_path)
tokenizer.pad_token_id = tokenizer.im_end_id

def format_example(example):
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    if example.get("input"):
        target = example["output"] + "<|im_end|>"
        context = prompt_input.format_map(
            dict(instruction=example["instruction"], input=example["input"])
        )
        example["context"] = context
        example["target"] = target
    else:
        target = example["output"] + "<|im_end|>"
        context = prompt_no_input.format_map(
            dict(instruction=example["instruction"])
        )
        example["context"] = context
        example["target"] = target
    return example

def preprocess(example):
    prompt = example["context"]
    target = example["target"]
    input_ids = tokenizer(
        prompt + target,
        return_tensors="pt",
        padding="longest",
        max_length=512,
        truncation=True,
    )
    seq_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding="longest",
        max_length=512,
        truncation=True,
    )
    input_ids_len = seq_ids.input_ids.ne(tokenizer.pad_token_id).sum().item()
    return {"input_ids": input_ids.input_ids[0], "seq_len": input_ids_len}

tokenized_datasets = dataset.map(function=format_example, num_proc=8, keep_in_memory=False)
tokenized_datasets = tokenized_datasets.map(function=preprocess, num_proc=8, keep_in_memory=False).shuffle(23333)
my_datasets = tokenized_datasets.train_test_split(test_size=4096)
def data_collator(features):
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)+1
    input_ids = []
    labels_list = []

    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len =  feature["seq_len"]
        labels = [-100]*seq_len + ids[seq_len:] + [-100]*(longest - ids_l)
        ids = ids + [tokenizer.im_end_id] * (longest - ids_l)
        ids = torch.LongTensor(ids)

        labels_list.append(torch.LongTensor(labels))
        input_ids.append(ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)

    return {"input_ids": input_ids, "labels": labels}

# 训练
model = QWenLMHeadModel.from_pretrained(sftArguments.model_path)
model_size = sum(t.numel() for t in model.parameters())
print(f"Qwen size: {model_size / 1000**2:.2f}M parameters")

args = TrainingArguments(
    output_dir=sftArguments.model_save_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.1,
    warmup_steps=0,
    learning_rate=6e-5,
    ddp_find_unused_parameters=False,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    optim="adamw_torch",
    remove_unused_columns=False,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    log_level="info",
    logging_first_step=True,
)


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=my_datasets["train"],
    eval_dataset=my_datasets["test"],
)

trainer.train()


eval_results = trainer.evaluate()
print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")
loss_log = pd.DataFrame(trainer.state.log_history)

trainer.save_model(sftArguments.model_save_dir)














