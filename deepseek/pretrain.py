import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, Trainer, \
    DataCollatorForLanguageModeling

tokenizer = AutoTokenizer.from_pretrained("/openbayes/input/input0/DeepSeek-R1-Distill-Qwen-1.5B")

# 加载数据集
# streaming是有惰性的，不会立即执行，只有在需要的时候才会执行
data_files = {
    "train": "/openbayes/home/opentest/datasets/Sky_train.jsonl",
    "test": "/openbayes/home/opentest/datasets/Sky_test.jsonl"
}
dataset = load_dataset("json",data_files=data_files)
print(dataset)

def tokenize_to_id(samples):
    batch_text = samples["text"]
    outputs = tokenizer(batch_text, truncation=True, max_length=512, return_attention_mask=False)
    input_ids = [np.array(item) for item in outputs["input_ids"]]
    return {"input_ids": input_ids}
train_dataset = dataset["train"]
test_dataset = dataset["test"]

train_dataset = train_dataset.map(tokenize_to_id, batched=True)
test_dataset = test_dataset.map(tokenize_to_id, batched=True)
print(train_dataset)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir='./results',            # 保存模型的目录
    num_train_epochs=3,                # 训练轮数
    per_device_train_batch_size=1,     # 训练批次大小
    per_device_eval_batch_size=1,      # 验证批次大小
    warmup_steps=500,                  # 训练的预热步数
    weight_decay=0.01,                 # 权重衰减
    logging_dir='./logs',              # 日志目录
    logging_steps=10,                  # 多少步记录一次日志
    evaluation_strategy='steps',       # 评估策略
    save_strategy='epoch',             # 保存策略
    learning_rate=1e-4,                # 学习率
    eval_steps=100                     # 多少步评估一次
)

# 定义模型
deepseek = AutoModelForCausalLM.from_pretrained("/openbayes/input/input0/DeepSeek-R1-Distill-Qwen-1.5B")
for name, param in deepseek.named_parameters():
    if name == "lm_head.weight":
        continue
    param.requires_grad = False

# 开始训练!
trainer = Trainer(
    model=deepseek,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

trainer.train()













































