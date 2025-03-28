import argparse
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/openbayes/input/input0/DeepSeek-R1-Distill-Qwen-1.5B")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default="/openbayes/home/opentest/datasets/Sky_train.jsonl" ,required=False, help='Directory including the configuration file')
    return parser.parse_args()

def load_json_to_token(file):
    train_data_name = "/openbayes/home/opentest/datasets/pre_train.pt"
    eval_data_name = "/openbayes/home/opentest/datasets/pre_test.pt"
    with open(file, 'r', encoding='utf-8') as f:
        examples_list = json.load(f)
    result = []
    for examples in tqdm(examples_list, desc="Processing examples"):
        input = tokenizer(examples["boy"], truncation=True, max_length=512, return_attention_mask=False)["input_ids"]
        label = input
        input = torch.LongTensor(input)
        label = torch.LongTensor(label)
        result.append({"input_ids": input,"label": label})
    torch.save(result[1000:], train_data_name)
    torch.save(result[1000:], eval_data_name)
    return torch.load(train_data_name), torch.load(eval_data_name)




if __name__ == '__main__':
    args = parse_args()
    train_file = "/openbayes/home/opentest/data/train_chat_dataset.json"
    load_json_to_token(train_file)
