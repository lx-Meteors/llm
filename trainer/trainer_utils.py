import os
import random
from logging import Logger

import torch.distributed as dist

import numpy as np
import torch
from transformers import AutoTokenizer

from models.model_meteor import MeteorForCausalLM


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def init_distributed_mode():
    # 单卡
    if int(os.environ.get("RANK"), -1) == -1:
        return 0
    # 多卡
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def meteor_checkpoint():
    # todo: checkpoint
    return None

def init_model(meteor_config, from_weight='pretrain', tokenizer_path='../models', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MeteorForCausalLM(meteor_config)

    if from_weight != 'none':
        moe_suffix = '_moe' if meteor_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{meteor_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    Logger(f'所加载Model可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model.to(device), tokenizer












