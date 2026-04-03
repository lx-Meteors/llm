# miniMind_Study

从 0 到 1 构建属于自己的 LLM（MeteorMind 学习工程）。

## 项目目标

- 按阶段学习并实现完整 LLM 流程：Pretrain -> SFT -> Alignment（PPO/DPO/GRPO）
- 保持“代码实现 + 学习笔记”双轨推进，避免只跑通不沉淀
- 在小规模可复现设置下先打通训练链路，再逐步扩展

## 当前进度

- 已完成 Pretrain 主流程（数据集、训练入口、优化器与保存恢复）
- 已建立阶段化笔记体系（见 `notes/`）
- SFT 与 Alignment 笔记模板已就绪，待进入实现阶段

## 仓库结构

```text
datasets/
	pretrain_dataset.py      # 预训练数据集读取与 Causal LM 对齐
	pretrain_hq.jsonl        # 预训练语料
	sft_mini_512.jsonl       # SFT 示例数据

models/
	model_meteor.py          # Meteor 模型实现（Attention、RoPE、FFN、Block）

trainer/
	train_pretrain.py        # 预训练入口（DDP、AMP、梯度累积、保存）
	trainer_utils.py         # 训练工具（学习率、checkpoint、seed、初始化）

scripts/
	train_tokenizer.py       # ByteLevel BPE tokenizer 训练脚本

notes/
	README.md
	learning_journal.md
	stage_pretrain.md
	stage_sft.md
	stage_alignment_rl.md
	pretrain_tips_digest.md  # 从代码 Tips 提炼出的实现注释笔记
```

## Pretrain 训练链路

1. 使用 `scripts/train_tokenizer.py` 训练并保存 tokenizer
2. `datasets/pretrain_dataset.py` 读取 jsonl 文本并构造 `(input_ids, labels, loss_mask)`
3. `trainer/train_pretrain.py` 执行训练：
	 - 混合精度（bfloat16/float16）
	 - 梯度累积与梯度裁剪
	 - 余弦学习率调度
	 - DDP 训练（可选）
	 - ckpt 与 resume 状态保存

## 快速开始

### 1) 训练 tokenizer

```bash
cd scripts
python train_tokenizer.py
```

或使用后台脚本：

```bash
bash run.sh
```

### 2) 运行 pretrain

```bash
python -m trainer.train_pretrain \
	--epochs 1 \
	--batch_size 32 \
	--max_seq_len 512 \
	--learning_rate 1e-4 \
	--data_path ../datasets/pretrain_hq.jsonl
```

多卡可使用：

```bash
torchrun --nproc_per_node=4 -m trainer.train_pretrain --epochs 1
```

## 训练产物

- 权重默认保存到 `out/`
- 训练状态与 resume 信息保存到 `checkpoints/`

## 已知事项

- 目前 `train_pretrain.py` 与 `trainer_utils.py` 中模型/配置默认从 `test/ok.py` 导入，属于临时占位；后续建议切回 `models/model_meteor.py`
- `models/model_meteor.py` 中保留了多处 Tips/TODO，已整理到 `notes/pretrain_tips_digest.md`

## 学习笔记使用方式

- 每次学习后在 `notes/learning_journal.md` 追加一条日志
- 将稳定结论沉淀到阶段文档（如 `notes/stage_pretrain.md`）
- 遇到实现疑问优先记录到 `notes/pretrain_tips_digest.md`，并逐条补全
