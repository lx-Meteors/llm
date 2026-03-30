# Stage: Pretrain

## 1. Goal

Learn general language modeling capability from large-scale text.

## 2. Core concepts

- Data quality and coverage
- Tokenizer design and vocabulary
- Causal language modeling objective
- Scaling laws: params, data, compute
- Optimization stability

## 3. Key formulas

Autoregressive objective:

$$
\max_\theta \sum_t \log p_\theta(x_t \mid x_{<t})
$$

Cross-entropy loss:

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(y_i)
$$

## 4. Repo mapping

- Dataset pipeline: `datasets/pretrain_dataset.py`
- Training entry: `trainer/train_pretrain.py`
- Model definition: `models/model_meteor.py`
- Tokenizer script: `scripts/train_tokenizer.py`

## 4.1 Current implementation checklist (done)

- [x] Tokenizer training script (ByteLevel BPE)
- [x] Pretrain dataset jsonl loading and tokenization
- [x] Causal LM shift alignment (`x_{<t}` -> `x_t`)
- [x] Mixed precision training (bf16/fp16)
- [x] Gradient accumulation + grad clipping
- [x] Cosine LR schedule
- [x] DDP support and resume checkpoint

## 4.2 Runtime flow (this repo)

1. Build tokenizer with `scripts/train_tokenizer.py`.
2. Load dataset in `PretrainDataset` and return `input_ids/labels/loss_mask`.
3. Build model and optimizer in `trainer/train_pretrain.py`.
4. Train with autocast + scaler + accumulation.
5. Save fp16 weights and full resume states periodically.

## 4.3 Training sanity checks

- Loss should be finite and generally decrease in early steps.
- Grad norm should stay bounded after clipping.
- Effective batch size = `batch_size * gradient_accumulation_steps * world_size`.
- Resume run should continue from saved epoch/step without data overlap surprises.
- Tokenizer special tokens must keep stable ids across runs.

## 5. What to record each time

- Data sample and cleaning rule changed
- Context length and batch setup
- Learning rate schedule and warmup
- Loss curve shape and anomalies
- Throughput and memory usage

## 6. Common pitfalls

- Training instability from overly high learning rate
- Poor tokenizer coverage causes fragmented tokens
- Data leakage between train and validation split
- Ignoring gradient norm and exploding updates

## 7. Next experiments

- [ ] Compare tokenizer vocab sizes
- [ ] Compare sequence length impact on throughput/loss
- [ ] Add a minimal validation perplexity tracker

## 8. Tips digest entry

- See `notes/pretrain_tips_digest.md` for the code-level Tips and concise explanations.
