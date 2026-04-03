import os
import sys

from dataset.sft_dataset import SFTDataset
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import os
import time
import warnings
from contextlib import nullcontext
from logging import Logger
import torch.nn.functional as F
import torch.distributed as dist
import torch.cuda
import wandb
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader

from models.model_meteor import MeteorConfig
from dataset.pretrain_dataset import PretrainDataset
from trainer.trainer_utils import init_distributed_mode, setup_seed, meteor_checkpoint, is_main_process, init_model, \
    SkipBatchSampler, get_learning_rate

warnings.filterwarnings('ignore')

def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    # 为什么学生的是logits而教师的是概率分布？因为KL散度的计算需要教师模型的概率分布作为目标，而学生模型输出的是未经归一化的logits。通过对教师模型的logits应用softmax函数，我们可以得到教师模型的概率分布，然后再计算学生模型的log_softmax与教师模型的softmax之间的KL散度。这种方式可以更稳定地计算损失，避免数值不稳定的问题。
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return kl * (temperature ** 2)  # 缩放损失以补偿温度带来的梯度变化


def train_epoch(args, model, optimizer, teacher_model, autocast_ctx, scaler, meteor_config_student, epoch, loader, iters, start_step=0, wandb=None, alpha=0.0, temperature=1.0):
    start_time = time.time()

    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (input_ids, labels, loss_mask) in enumerate(loader, start=start_step+1):
        # 输入数据
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        loss_mask = loss_mask.to(args.device)
        # 学习率
        learning_rate = get_learning_rate(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 动态更新优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate


        # 前向传播（学生模型）
        with autocast_ctx:
            res = model(input_ids)
            student_logits = res.logits  # [batch_size, seq_len, vocab_size]

        # 前向传播（教师模型）
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits  # [batch_size, seq_len, vocab_size]
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[:, :, :vocab_size_student]  # 对齐教师和学生的词表维度

        # 计算损失
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=0,
            reduction='none'
        )
        ce_loss = torch.sum(ce_loss * loss_mask_flat) / loss_mask_flat.sum()
        if meteor_config_student.use_moe:
            ce_loss += res.aux_loss

        if teacher_model is not None:
            # 计算KL散度损失
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 总损失 = alpha * CE损失 + (1 - alpha) * KL散度损失
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.gradient_accumulation_steps
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
            moe_suffix = '_moe' if meteor_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{meteor_config_student.hidden_size}{moe_suffix}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            meteor_checkpoint(meteor_config_student, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()

        del input_ids, labels, loss_mask, output, loss




def main():
    parser = argparse.ArgumentParser(description="Meteor-LLM Knowledge Distillation")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保持目录")
    parser.add_argument("--save_weight", default="full_distill", help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累计步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument('--student_hidden_size', default=512, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量")
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=16, type=int, help="教师模型隐藏层数量")
    parser.add_argument("--max_seq_len", type=int, default=512, help="训练最大截断长度")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径")
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="学生模型基于哪个权重")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师模型基于哪个权重")
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0）")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1], help="是否续训（0=否，1=是）")
    parser.add_argument("--use_wandb", type=int, default=1, choices=[0, 1], help="是否使用wandb（0=否，1=是）")
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
    parser.add_argument("--wandb_project", type=str, default="Meteor-Full-SFT", help="wandb项目名")
    args = parser.parse_args()

    # 1. 初始化环境和随机种子
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 2. 配置目录、模型参数、检查ckp
    os.makedirs(args.save_dir, exist_ok=True)
    meteor_config_student = MeteorConfig(hidden_size=args.student_hidden_size, num_hidden_layers=args.student_num_layers, use_moe=bool(args.use_moe))
    meteor_config_teacher = MeteorConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layers, use_moe=bool(args.use_moe))

    ckp_data = meteor_checkpoint(meteor_config_student, weight=args.save_weight,
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

    # 5. 定义学生和教师模型
    model, tokenizer = init_model(meteor_config_student, args.from_student_weight, device=args.device)
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')

    teacher_model, _ = init_model(meteor_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model.eval()  # 教师模型只进行前向传播
    teacher_model.requires_grad_(False)
    Logger(f'教师模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')

    # 加载数据、建立索引
    train_dataset = SFTDataset(args.data_path, tokenizer, args.max_seq_len)
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
            train_epoch(args, model, optimizer, autocast_ctx, scaler, teacher_model, meteor_config_student, epoch, loader, len(loader) + start_step + 1, start_step, wandb, args.alpha, args.temperature)
        else:
            loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(args, model, optimizer, autocast_ctx, scaler, teacher_model, meteor_config_student, epoch, loader, len(loader), 0, wandb, args.alpha, args.temperature)

if __name__ == '__main__':
    main()

