import json

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512):
        super().__init__()
        self.samples = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len


    def load_data(self, data_path):
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                samples.append(data)
            return samples


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]['text']
        # 构建输入token
        encoding = self.tokenizer(
            sample,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)
        # shift 输入去掉最后一个token labels去掉第一个token（这样符合当前词预测下一个词）
        input_ids = torch.tensor(input_ids[:-1], dtype=torch.long)
        labels = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return input_ids, labels, loss_mask


































