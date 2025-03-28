from transformers import AutoModelForCausalLM
import torch
from process_data import load_json_to_token
from dataset import MyDataSet
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
# ==========================处理数据========================================
train_file = "/openbayes/home/opentest/data/train_chat_dataset.json"
# 1. 处理数据
train_data, test_data = load_json_to_token(train_file)
# 2. 加载数据
dataset = MyDataSet(train_data)
# 3. 将加载的数据转为可迭代的批量数据加载器
dataloader = DataLoader(dataset, batch_size=1)

# ==========================加载模型=======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. 加载模型
deepseek = AutoModelForCausalLM.from_pretrained("/openbayes/input/input0/DeepSeek-R1-Distill-Qwen-1.5B").to(device)
# 1.1 全参数训练用显存太大了，只是为了学习测试，只训练其lm_head
for name, param in deepseek.named_parameters():
    print(name, param.shape)
    if name == "lm_head.weight":
        continue
    param.requires_grad = False
# 2. 定义优化器
optimizer = torch.optim.AdamW(deepseek.parameters(), lr=0.0001, betas=(0.9, 0.95), weight_decay=0.1)
# 3. 定义损失函数
criterion = nn.CrossEntropyLoss()
# ========================开始训练=======================================
for epoch in range(1):
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        deepseek.train()
        input_ids, label = batch["input_ids"].to(device), batch["label"].to(device)
        optimizer.zero_grad()  # 清空梯度
        outputs = deepseek(input_ids)  # 前向传播
        shift_logits = outputs.logits[:, :-1, :].contiguous()  # 形状 [batch_size, seq_len-1, vocab_size]
        shift_labels = label[:, 1:].contiguous()  # 形状 [batch_size, seq_len-1]
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_loss += loss.item()



























