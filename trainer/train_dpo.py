import os
import sys

import wandb

from dataset.dpo_dataset import DPODataset
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from models.model_meteor import MeteorConfig
from dataset.pretrain_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, meteor_checkpoint, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')

def logits_to_log_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 将chosen和rejected数据分开
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    rejected_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    rejected_policy_log_probs = policy_log_probs[batch_size // 2:]

    pi_logratios = chosen_policy_log_probs - rejected_policy_log_probs
    ref_logratios = chosen_ref_log_probs - rejected_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(args, model, optimizer, ref_model, autocast_ctx, scaler, meteor_config, epoch, loader, iters, start_step=0, wandb=None, beta=0.1):
    start_time = time.time()

    for step, batch in enumerate(loader, start=start_step+1):
        # 输入数据
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        with autocast_ctx:
            with torch.no_grad():
                ref_output = ref_model(x)
                ref_logits = ref_output.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)

            loss = dpo_loss(policy_log_probs, ref_log_probs, mask, beta)
            loss = loss / args.gradient_accumulation_steps
        scaler.scale(loss).backward()




        if (step + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.gradient_accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')

            if wandb.run is not None: wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if meteor_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{meteor_config.hidden_size}{moe_suffix}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            meteor_checkpoint(meteor_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()

        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss



def main():
    parser = argparse.ArgumentParser(description="Meteor-LLM DPO")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保持目录")
    parser.add_argument("--save_weight", default="dpo", help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累计步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="隐藏层数量")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="训练最大截断长度")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径")
    parser.add_argument("--from_weight", type=str, default="pretrain", help="基于哪个权重训练，为none则从头开始")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1], help="是否续训（0=否，1=是）")
    parser.add_argument('--beta', default=0.1, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", type=int, default=1, choices=[0, 1], help="是否使用wandb（0=否，1=是）")
    parser.add_argument("--wandb_project", type=str, default="Meteor-Full-SFT", help="wandb项目名")
    args = parser.parse_args()

     # 1. 初始化环境和随机种子
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 2. 配置目录、模型参数、检查ckp
    os.makedirs(args.save_dir, exist_ok=True)
    meteor_config = MeteorConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                                 use_moe=bool(args.use_moe))
    ckp_data = meteor_checkpoint(meteor_config, weight=args.save_weight,
                                 save_dir='../checkpoints') if args.from_resume == 1 else None

    # 3. 设置混合精度
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if args.device == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # 4. 配置wandb
    if args.use_wandb and is_main_process():
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"Meteor-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 5. 定义模型、数据、优化器
    model, tokenizer = init_model(meteor_config, args.from_weight, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 初始化参考模型(ref_model冻结)
    ref_model, _ = init_model(meteor_config, args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

    # 加载数据、建立索引
    train_dataset = DPODataset(args.data_path, tokenizer, args.max_seq_len)
    # 对不同卡设置不同数据索引
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 6. 从ckpt恢复状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch, start_step = ckp_data['epoch'], ckp_data['step']

    # 7. DDP包模型
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # 8. 开始训练
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        # 第一个epoch且存在ckpt
        if epoch == start_epoch and start_step > 0:
            # 根据索引train_sampler进行划分batch_size，并跳过前start_step个batch数据
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_dataset)), args.batch_size, start_step + 1)
            # loader得到的就是数据，DataLoader通过batch_sampler的索引去train_dataset拿数据
            loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(args, model, optimizer, ref_model, autocast_ctx, scaler, meteor_config, epoch, loader, len(loader) + start_step + 1, start_step, wandb, args.beta)
        else:
            loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(args, model, optimizer, ref_model, autocast_ctx, scaler, meteor_config, epoch, loader, len(loader), 0, wandb, args.beta)

if __name__ == '__main__':
    main()
