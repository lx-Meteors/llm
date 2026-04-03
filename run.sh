#!/usr/bin/env bash
set -euo pipefail

# 用法：
# 1) bash run.sh tokenizer
# 2) bash run.sh pretrain
# 3) bash run.sh full_sft

# session
# 开启session：tmux new -s full_sft
# 隐藏session：Ctrl + b d
# 进入session：tmux attach -t full_sft

MODE="${1:-none}"

if [[ "$MODE" == "tokenizer" ]]; then
	# 训练 tokenizer（后台）
	# tail -f scripts/train_tokenizer.log
	cd scripts
	nohup python train_tokenizer.py > train_tokenizer.log 2>&1 &
	echo "Tokenizer training started in background."
elif [[ "$MODE" == "pretrain" ]]; then
	# 预训练（多卡）
	# 显式传 --master_port，避免默认 29500 端口冲突
	cd trainer
	CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}" \
	torchrun --nproc_per_node=4 --master_port="${MASTER_PORT:-29501}" \
	train_pretrain.py
elif [[ "$MODE" == "full_sft" ]]; then
	# 全量微调（多卡）
	# 显式传 --master_port，避免默认 29500 端口冲突
	cd trainer
	CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3}" \
	torchrun --nproc_per_node=2 --master_port="${MASTER_PORT:-29501}" \
	train_full_sft.py
else
	echo "Unknown mode: $MODE"
	echo "Usage: bash run.sh [tokenizer|pretrain|full_sft]"
	exit 1
fi

