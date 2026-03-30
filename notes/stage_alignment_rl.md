# Stage: Alignment (PPO, DPO, GRPO)

## 1. Goal

Align model behavior with human preferences and improve response quality/safety.

## 2. Method map

- PPO: policy gradient with KL constraint and reward model
- DPO: direct preference optimization without explicit reward model training loop
- GRPO: group-relative objective variant for preference optimization

## 3. Core objective sketches

PPO-style clipped objective:

$$
\mathcal{L}^{CLIP}(\theta)=\hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right]
$$

DPO-style preference objective (informal):

$$
\max_\theta\ \mathbb{E}_{(x,y^+,y^-)}\left[\log \sigma\left(\beta\left(\log \pi_\theta(y^+|x)-\log \pi_\theta(y^-|x)\right)-\Delta_{ref}\right)\right]
$$

## 4. What to record each time

- Preference data source and labeling quality
- Reward/preference signal definition
- KL or regularization settings
- Win-rate style pairwise evaluation
- Safety regressions and mitigation actions

## 5. Common pitfalls

- Reward hacking and style collapse
- Over-optimization leading to bland outputs
- Missing a stable reference model baseline
- No offline sanity checks before long runs

## 6. Suggested evaluation matrix

- Helpfulness
- Harmlessness
- Honesty
- Instruction adherence
- Length/verbosity control

## 7. Next experiments

- [ ] Define one fixed preference dataset slice for reproducible comparison
- [ ] Compare PPO vs DPO on same prompt set
- [ ] Add a short post-run regression checklist
