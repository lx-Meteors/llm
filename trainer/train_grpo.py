import os
import sys
import wandb
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import gc
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset.rlaif_dataset import RLAIFDataset
from transformers import AutoModel, AutoTokenizer
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
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, meteor_checkpoint, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')

def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    # 整合所有奖励函数计算总奖励
    def reasoning_model_reward(rewards):
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern or matches_pattern2:
                format_rewards.append(0.5)  # 满足格式要求的奖励为1
            else:
                format_rewards.append(0.0)  # 不满足格式要求的奖励为0
        rewards += torch.tensor(format_rewards, device=rewards.device)

        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            return reward
    
        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=rewards.device)
        return rewards

    rewards = torch.zeros(len(responses), device=reward_model.device)
    if args.reasoning == 1:  # type: ignore
        rewards = reasoning_model_reward(rewards)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)
        scale = 3.0

        for i in range(batch_size):
            for j in range(args.num_generations):  # type: ignore
                response_idx = i * args.num_generations + j  # type: ignore
                response = responses[response_idx]
                prompt = prompts[i]

                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]

                tmp_chat = messages + [{"role": "assistant", "content": response}]
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                score = max(min(score, scale), -scale)  # 将分数限制在[-scale, scale]范围内

                if args.reasoning == 1: # type: ignore
                    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                        answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                        answer_score = max(min(answer_score, scale), -scale)
                        score = score * 0.4 + answer_score * 0.6  # 综合考虑整体和answer部分的得分
                reward_model_scores.append(score)
        
        reward_model_scores = torch.tensor(reward_model_scores, device=reward_model.device)
        rewards += reward_model_scores
    return rewards





def train_epoch(args, model, optimizer, autocast_ctx, meteor_config, epoch, ref_model, reward_model, reward_tokenizer, loader, iters, start_step=0, wandb=None, beta=0.1):
    for step, batch in enumerate(loader, start=start_step+1):
        prompts = batch['prompt'] 
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False, # type: ignore
                                  padding_side="left", add_special_tokens=False).to(args.device) 

        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        with torch.no_grad():
            # DDP模型需要使用.module访问
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            # [B*num_gen, P+R]
            outputs = model_for_gen.generate(
                **prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                num_return_sequences=args.num_generations, pad_token_id=tokenizer.pad_token_id) # type: ignore
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # [B*num_gen, R]

        def get_per_token_logps(model, input_ids, n_keep):
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            logits = model(input_ids, logits_to_keep=n_keep+1).logits[:, :-1, :]  # [B*num_gen, R, V]
            pre_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1))
                return torch.stack(per_token_logps)

        per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B*num_gen, R]
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B*num_gen, R]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)  # type: ignore
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)  # [B*num_gen]  # [B*num_gen]

        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)  # [B*num_gen]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 标准化优势

        is_eos = completion_ids == tokenizer.eos_token_id  # [B*num_gen, R] # type: ignore
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1)-1, device=is_eos.device)  # [B*num_gen]
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()  # [B*num_gen, R]

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
        per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) - args.beta * per_token_kl) 
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean() / args.accumulation_steps  # scalar
        loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()  # type: ignore
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch: {epoch+1}, Step: {step}/{iters}, '
                   f'Actor Loss: {policy_loss_val:.6f}, Reward: {avg_reward_val:.6f}, '
                   f'Avg Response Len: {avg_len_val:.2f}, LR: {current_lr:.2e}')

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "reward": avg_reward_val,
                    "avg_response_len": avg_len_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr
                })
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

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask
        torch.cuda.empty_cache()
        gc.collect()



def main():
    parser = argparse.ArgumentParser(description="Meteor-LLM GRPO")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保持目录")
    parser.add_argument("--save_weight", default="grpo", help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="初始学习率")
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
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="训练数据路径")
    parser.add_argument("--num_generations", type=int, default=8, help="每个prompt生成的样本数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--from_weight", type=str, default="pretrain", help="基于哪个权重训练，为none则从头开始")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1], help="是否续训（0=否，1=是）")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--beta', default=0.02, type=float, help="KL中的beta参数")
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
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    # Policy模型
    model, tokenizer = init_model(meteor_config, base_weight, device=args.device)
    # Reference模型
    ref_model, _ = init_model(meteor_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Reward模型
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)

    # 加载数据、建立索引
    train_dataset = RLAIFDataset(args.data_path, tokenizer, args.max_seq_len)
    # 对不同卡设置不同数据索引
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    

    # 6. 从ckpt恢复状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
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
            train_epoch(args, model, optimizer, autocast_ctx, meteor_config, epoch, ref_model, reward_model, reward_tokenizer, loader, len(loader) + start_step + 1, start_step, wandb, args.beta)
        else:
            loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(args, model, optimizer, autocast_ctx, meteor_config, epoch, ref_model, reward_model, reward_tokenizer, loader, len(loader), 0, wandb, args.beta)

if __name__ == '__main__':
    main()
