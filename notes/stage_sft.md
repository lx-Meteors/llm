# Stage: SFT

## 1. Goal

Specialize the pretrained model to follow instructions and produce task-aligned outputs.

## 2. Core concepts

- Instruction data schema
- Prompt formatting and role tags
- Supervised objective over response tokens
- Curriculum and difficulty mix
- Overfitting vs generalization

## 3. Key formulas

SFT loss on target response tokens:

$$
\mathcal{L}_{SFT} = -\frac{1}{M}\sum_{j=1}^{M}\log p_\theta(r_j \mid x, r_{<j})
$$

## 4. Repo mapping

- SFT data example: `datasets/sft_mini_512.jsonl`
- Model definition: `models/model_meteor.py`
- Training utilities: `trainer/trainer_utils.py`

## 5. What to record each time

- Prompt template version
- Data dedup/filtering strategy
- Effective batch size and max length
- Validation samples and failure patterns
- Instruction-following quality changes

## 6. Common pitfalls

- Training on noisy instruction/answer pairs
- Mismatch between training and inference prompt template
- Evaluating only loss but not behavior
- Catastrophic forgetting from narrow SFT data

## 7. Next experiments

- [ ] Build a tiny benchmark set for instruction following
- [ ] Compare with/without system prompt during training
- [ ] Add behavior-based evaluation notes
