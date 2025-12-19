import argparse
import os
from contextlib import nullcontext
from distutils import dist

import torch.cuda
import wandb
from models.model_meteor import MeteorConfig
from trainer.trainer_utils import init_distributed_mode, setup_seed, meteor_checkpoint, is_main_process, init_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Meteor-LLM Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保持目录")
    parser.add_argument("--save_weight", default="pretrain", help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累计步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="隐藏层数量")
    parser.add_argument("--max_seq_len", type=int, default=512, help="训练最大截断长度")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0,1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../datasets/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument("--from_weight", type=str, default="none", help="基于哪个权重训练，为none则从头开始")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0,1], help="是否续训（0=否，1=是）")
    parser.add_argument("--use_wandb", type=int, default=1, choices=[0,1], help="是否使用wandb（0=否，1=是）")
    parser.add_argument("--wandb_project", type=str, default="Meteor-LLM Pretrain", help="wandb项目名")
    args = parser.parse_args()

    # 1. 初始化环境和随机种子
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0)))

    # 2. 配置目录、模型参数、检查ckp
    os.makedirs(args.save_dir, exist_ok=True)
    meteor_config = MeteorConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = meteor_checkpoint()

    # 3. 设置混合精度
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if args.device == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # 4. 配置wandb
    if args.use_wandb and is_main_process():
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"Meteor-LLM-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 5. 定义模型、数据、优化器
    model, tokenizer = init_model(meteor_config, args.from_weight, device=args.device)


































