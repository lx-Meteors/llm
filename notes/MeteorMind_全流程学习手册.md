# MeteorMind 全流程学习手册（从 0 到 1）

版本：v1.0  
适用仓库：当前 MeteorMind 学习工程（llm）

---

## 第 1 章 这本手册怎么用

### 1.1 目标读者

- 完全从零开始学大模型训练的人
- 已经会跑脚本，但不知道每一步为什么这么写的人
- 想把 Pretrain -> SFT -> DPO/GRPO -> 蒸馏串起来的人

### 1.2 学习目标

读完后你应该能做到：

- 说清楚每个阶段在优化什么目标
- 看懂本仓库每个核心脚本在做什么
- 能解释关键公式和代码实现之间的对应关系
- 能自己定位训练过程中的常见问题

### 1.3 推荐阅读顺序

1. 先读第 2 章项目全景
2. 再读第 3 到第 6 章（Tokenizer、数据、模型、Pretrain）
3. 再读第 7 到第 10 章（SFT、DPO、GRPO、蒸馏）
4. 最后看第 11 到第 14 章（推理、工程化、调参、排错）

---

## 第 2 章 项目全景图

### 2.1 目录结构与职责

- dataset/
  - pretrain_dataset.py：预训练数据集
  - sft_dataset.py：SFT 数据集
  - dpo_dataset.py：偏好对数据集
  - rlaif_dataset.py：GRPO 的 prompt 数据集
- models/
  - model_meteor.py：核心 Transformer 模型
  - model_lora.py：LoRA 注入和加载
- trainer/
  - train_pretrain.py：预训练
  - train_full_sft.py：监督微调
  - train_dpo.py：DPO
  - train_grpo.py：GRPO
  - train_distillation.py：知识蒸馏
  - trainer_utils.py：训练公共工具
- scripts/
  - train_tokenizer.py：训练和评估 tokenizer
- eval_llm.py：推理和交互
- notes/：学习笔记

### 2.2 一句话理解整条训练链

- Pretrain：学会“说人话”
- SFT：学会“按指令说话”
- DPO/GRPO：学会“说得更符合偏好”
- Distillation：把大模型能力压缩给小模型

### 2.3 统一训练抽象

几乎所有训练都可以抽象为：

1. 数据输入
2. 模型前向得到 logits
3. 计算 loss
4. 反向传播更新参数
5. 周期记录与保存 checkpoint

---

## 第 3 章 Tokenizer 基础与实现

### 3.1 为什么 tokenizer 这么重要

模型看不懂“字”，只看得懂“id”。Tokenizer 决定了文本被切成什么粒度，直接影响：

- 序列长度
- 训练效率
- 表达能力

### 3.2 本项目 tokenizer 方案

使用 ByteLevel BPE（scripts/train_tokenizer.py）：

- vocab_size = 6400
- special tokens：
  - <|endoftext|>
  - <|im_start|>
  - <|im_end|>

并构造了 chat_template，支持角色对话格式。

### 3.3 关键检查点

- special token id 是否稳定
- tokenizer.json 与 tokenizer_config.json 是否配套
- chat_template 是否与训练/推理使用方式一致

### 3.4 易懂类比

Tokenizer 就像“字典+切词规则”：

- 字典太小：很多词被拆得很碎，句子变长
- 字典太大：参数和训练数据需求上升

---

## 第 4 章 数据管线（Dataset）

### 4.1 PretrainDataset

文件：dataset/pretrain_dataset.py

输入：每行 json，字段 text。

输出：

- X = input_ids[:-1]
- Y = input_ids[1:]
- loss_mask = 非 pad 位置

核心思想：下一 token 预测。

### 4.2 SFTDataset

文件：dataset/sft_dataset.py

核心做法：

1. 用 apply_chat_template 把多轮对话拼成模型输入
2. 动态构建 loss_mask，只对 assistant 回答区域计算 loss

为什么只算 assistant 的损失：

- user/system 内容是条件，不是模型要“背诵”的目标

### 4.3 DPODataset

文件：dataset/dpo_dataset.py

每个样本包含 chosen 和 rejected 两个回答。

输出张量：

- chosen 的 x/y/mask
- rejected 的 x/y/mask

本质：让模型更偏向 chosen，远离 rejected。

### 4.4 RLAIFDataset

文件：dataset/rlaif_dataset.py

输出：

- prompt：用于在线采样
- answer：参考答案（当前 GRPO 主流程中主要用 prompt）

### 4.5 新手常见误区

- 误把所有 token 都算 loss，导致模型学会复读 prompt
- chat_template 训练和推理不一致，导致效果明显下降
- mask 与 shift（X/Y）错位，loss 无意义

---

## 第 5 章 模型结构（Meteor）

文件：models/model_meteor.py

### 5.1 总体结构

MeteorForCausalLM = MeteorModel + lm_head

MeteorModel 由以下组成：

- embedding
- 多层 MeteorBlock
- RMSNorm

MeteorBlock 内部：

1. Attention
2. FeedForward（SwiGLU 风格）
3. 残差连接 + Pre-Norm

### 5.2 自回归目标

$$
\max_\theta \sum_t \log p_\theta(x_t \mid x_{<t})
$$

对应代码上就是 shift 后的交叉熵。

### 5.3 注意力公式

$$
\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

其中 M 是 causal mask，保证当前位置看不到未来。

### 5.4 RoPE（旋转位置编码）

直觉：把 q/k 的每两个维度当二维向量旋转，旋转角随位置变化。

$$
\omega_i = \theta^{-2i/d},\quad \phi_{t,i}=t\cdot\omega_i
$$

为什么好用：

- 对长上下文外推更友好
- 在相对位置建模上更自然

### 5.5 RMSNorm

$$
\text{RMSNorm}(x)=\gamma \cdot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}
$$

相比 LayerNorm，RMSNorm 不做均值中心化，计算更简洁。

### 5.6 FeedForward（SwiGLU 变体）

$$
\text{FFN}(x)=W_d(\text{silu}(W_gx) \odot W_ux)
$$

解释：一条分支像“门”，一条分支像“内容”，乘起来后信息控制更细。

### 5.7 GQA/MQA 相关实现

通过 num_key_value_heads 控制 KV 头数，再用 repeat_kv 对齐到 query 头数，降低 KV 计算和显存压力。

---

## 第 6 章 Pretrain（预训练）

文件：trainer/train_pretrain.py

### 6.1 目标

用大规模文本学习通用语言建模能力。

### 6.2 Loss 公式

逐 token 交叉熵并按 mask 归一：

$$
\mathcal{L}=\frac{\sum_{b,t} m_{b,t}\cdot \text{CE}(\hat y_{b,t}, y_{b,t})}{\sum_{b,t} m_{b,t}}
$$

### 6.3 训练关键机制

- 混合精度（bf16/fp16）
- 梯度累积
- 梯度裁剪
- 动态学习率（余弦）
- DDP 分布式训练
- checkpoint 续训

### 6.4 有效 batch 大小

$$
\text{Effective Batch} = \text{batch\_size} \times \text{gradient\_accumulation\_steps} \times \text{world\_size}
$$

这行是你调参时最重要的换算关系之一。

### 6.5 新手易懂解释

- 梯度累积：显存不够时“攒几步再更新”
- 梯度裁剪：防止一步走太猛把模型训崩
- 余弦学习率：先大步找方向，后小步打磨

---

## 第 7 章 SFT（监督微调）

文件：trainer/train_full_sft.py

### 7.1 目标

让模型从“会说话”变成“会按指令说话”。

### 7.2 核心变化

和 Pretrain 相比，SFT 最大变化是 loss_mask：

- Pretrain：几乎全句都参与
- SFT：主要只让 assistant 回复区域参与

### 7.3 SFT 公式

$$
\mathcal{L}_{SFT} = -\frac{1}{M}\sum_{j=1}^{M}\log p_\theta(r_j \mid x, r_{<j})
$$

其中 x 是指令上下文，r 是回复 token。

### 7.4 训练策略建议

- 小学习率
- 保持模板一致（训练和推理）
- 加少量行为评测样例，不能只看 loss

---

## 第 8 章 DPO（偏好优化）

文件：trainer/train_dpo.py

### 8.1 DPO 的直观理解

不显式训练奖励模型，直接用偏好对（chosen, rejected）训练策略。

### 8.2 目标形式（简化）

$$
\max_\theta \; \mathbb{E}_{(x,y^+,y^-)}\left[\log \sigma\left(\beta\left((\log\pi_\theta(y^+|x)-\log\pi_\theta(y^-|x))-(\log\pi_{ref}(y^+|x)-\log\pi_{ref}(y^-|x))\right)\right)\right]
$$

解释：

- 希望策略模型比参考模型更“偏好 chosen”
- beta 控制偏好拉开的力度

### 8.3 代码映射

- logits_to_log_probs：取出标签 token 的对数概率
- dpo_loss：分别聚合 chosen/rejected，再做 log-ratio 差

### 8.4 实战注意

- 偏好数据质量比数量更关键
- chosen/rejected 的模板必须一致
- beta 过大可能训练不稳

---

## 第 9 章 GRPO（组相对策略优化）

文件：trainer/train_grpo.py

### 9.1 GRPO 一句话

同一个 prompt 采样多条回答，在组内比较谁更好，然后按相对优势更新策略。

### 9.2 组内优势计算

设同一组 reward 为 r_i：

$$
\tilde r_i = \frac{r_i - \mu_{group}}{\sigma_{group}+\epsilon}
$$

代码中再做了全局标准化，提高训练稳定性。

### 9.3 KL 正则项

代码使用的每 token KL 近似形式：

$$
\text{KL}_{token}=e^{\Delta}-\Delta-1,\quad \Delta=\log\pi_{ref}-\log\pi_\theta
$$

性质：非负，适合作为惩罚项。

### 9.4 per-token 损失（代码对应）

核心表达：

$$
\mathcal{L}_{token} = -\left(\exp(\log p - \text{sg}(\log p))\cdot A - \beta\cdot \text{KL}_{token}\right)
$$

其中 sg 是 stop-gradient。

易懂解释：

- 前项鼓励高优势回答
- 后项约束不要偏离参考模型太远

### 9.5 EOS 掩码

构建 completion_mask，只在 EOS 之前位置计算损失，避免把无效 token 算进去。

### 9.6 奖励来源

本实现是混合奖励：

- 规则奖励（格式标签、标记完整性）
- reward model 打分
- reasoning 模式下对 answer 子串再打一次分并加权

---

## 第 10 章 Distillation（知识蒸馏）

文件：trainer/train_distillation.py

### 10.1 目标

把教师模型输出分布“教给”学生模型。

### 10.2 蒸馏损失

KL 蒸馏项：

$$
\mathcal{L}_{KD}=T^2\cdot \text{KL}\left(\log\text{softmax}(z_s/T),\;\text{softmax}(z_t/T)\right)
$$

总损失：

$$
\mathcal{L}=\alpha\mathcal{L}_{CE} + (1-\alpha)\mathcal{L}_{KD}
$$

### 10.3 温度 T 的直觉

- T 越高，教师分布越“平”，暗知识更多
- T 太高会过平，监督变弱

### 10.4 alpha 的直觉

- alpha 大：更像普通监督训练
- alpha 小：更信任教师软标签

---

## 第 11 章 推理与评测

文件：eval_llm.py

### 11.1 推理流程

1. 加载 tokenizer 和模型
2. 构造对话模板
3. generate 采样（temperature/top_p）
4. decode 输出

### 11.2 常见采样参数

- temperature：随机性
- top_p：核采样截断
- max_new_tokens：单轮最大输出长度

### 11.3 评测建议

至少维护三类测试集：

- 指令遵循
- 事实正确性
- 长文本稳定性

---

## 第 12 章 工程化能力（你项目里已经有的硬核点）

### 12.1 DDP 分布式

trainer_utils.py 的 init_distributed_mode 负责：

- 初始化进程组
- 绑定 local_rank 到 GPU

### 12.2 混合精度

- bf16：通常更稳
- fp16：更省显存，但对 scaler 依赖更强

### 12.3 Checkpoint 体系

meteor_checkpoint 同时保存：

- 轻量推理权重
- 恢复训练所需完整状态（optimizer、epoch、step 等）

并支持 world_size 变化时 step 的换算。

### 12.4 SkipBatchSampler 续训跳步

用于断点恢复时跳过已训练 batch，避免重复学习同一段数据。

---

## 第 13 章 参数与调参方法

### 13.1 一组通用调参顺序

1. 先固定数据与模板
2. 再调学习率
3. 再调有效 batch
4. 最后调正则（dropout、beta、grad_clip）

### 13.2 你这个项目里最常改的参数

- hidden_size
- num_hidden_layers
- max_seq_len
- learning_rate
- gradient_accumulation_steps
- beta（DPO/GRPO）
- temperature（蒸馏和推理都有）

### 13.3 经验法则

- loss 不降：先查数据和 mask，再查学习率
- loss 振荡：降学习率或加大 batch 等效规模
- 生成变短：检查 eos 处理与长度惩罚
- 回答模板错位：优先核对 chat_template

---

## 第 14 章 常见坑位与排错手册

下面是从当前代码阅读中总结的高价值排错点。

### 14.1 路径命名不一致

仓库有 dataset/ 目录，但部分说明和脚本仍出现 datasets/。这会导致文件找不到。

排查方式：

- 先统一所有数据路径指向 dataset/
- 运行前用 ls 或打印绝对路径确认

### 14.2 变量名/函数名不一致

典型症状：运行时报 NameError 或参数不存在。

排查方式：

- 重点检查 gradient_accumulation_steps 与 accumulation_steps
- 检查 get_lr 与 get_learning_rate 的导入和定义一致性

### 14.3 训练循环中的未定义变量

典型症状：del 阶段或日志阶段报未定义。

排查方式：

- 全局搜索每个变量首次定义位置
- 尤其关注输出变量名 output/outputs 与 ref_output/ref_outputs

### 14.4 Train/Eval 模板不一致

表现：训练 loss 还行，但推理质量差、角色错乱。

排查方式：

- 把训练输入样例和推理输入样例打印出来逐字比对

### 14.5 DDP 相关问题

- 不同卡数据重复：检查 sampler.set_epoch(epoch)
- 保存多份 checkpoint：只在主进程保存

---

## 第 15 章 公式总览（速查）

### 15.1 自回归语言建模

$$
\max_\theta \sum_t \log p_\theta(x_t|x_{<t})
$$

### 15.2 交叉熵

$$
\mathcal{L}_{CE} = -\sum_i y_i\log \hat y_i
$$

### 15.3 注意力

$$
\text{softmax}(QK^\top/\sqrt{d_k})V
$$

### 15.4 RMSNorm

$$
\gamma \cdot x / \sqrt{\text{mean}(x^2)+\epsilon}
$$

### 15.5 DPO 核心差分

$$
(\log\pi_\theta^+ - \log\pi_\theta^-) - (\log\pi_{ref}^+ - \log\pi_{ref}^-)
$$

### 15.6 GRPO 组内归一化

$$
A_i = (r_i-\mu_{group})/(\sigma_{group}+\epsilon)
$$

### 15.7 蒸馏 KL

$$
T^2\cdot \text{KL}(\log\text{softmax}(z_s/T),\text{softmax}(z_t/T))
$$

---

## 第 16 章 给初学者的完整实战路线

### 16.1 第 1 周：把链路跑通

- 跑 tokenizer
- 跑最小 pretrain
- 能在 eval_llm.py 看到合理输出

### 16.2 第 2 周：做稳 SFT

- 只改一个变量做对照实验
- 固定一组验证 prompts 做行为追踪

### 16.3 第 3 周：做偏好对齐

- 先 DPO，再 GRPO
- 对照比较回答质量、稳定性、长度分布

### 16.4 第 4 周：做蒸馏和工程收口

- 完成 teacher -> student 蒸馏
- 记录吞吐、显存、质量三项对比

---

## 第 17 章 建议你继续补充的内容

为了让这份手册更像“可教学教材”，后续建议追加：

1. 每章附 3 道练习题（概念题、公式题、代码定位题）
2. 增加一份最小可复现实验清单（参数、seed、结果）
3. 增加一份术语表（如 GQA、RoPE、KL、advantage）
4. 增加阶段里程碑表（达到什么现象才算通过）

---

## 附录 A 本仓库核心文件导航

- 训练工具：trainer/trainer_utils.py
- 预训练：trainer/train_pretrain.py
- 全量 SFT：trainer/train_full_sft.py
- DPO：trainer/train_dpo.py
- GRPO：trainer/train_grpo.py
- 蒸馏：trainer/train_distillation.py
- 模型：models/model_meteor.py
- 推理：eval_llm.py

---

## 附录 B 学习方法建议

当你看不懂一段训练代码时，按这个顺序拆：

1. 输入张量 shape 是什么
2. 前向输出 shape 是什么
3. loss 用了哪些位置（mask）
4. 优化器何时 step
5. 日志打印的指标是否对应核心目标

这套方法可以解决 80% 的新手困惑。

---

如果你是第一次接触大模型训练，不用追求一口气全懂。按章节推进，每章只掌握一句话目标、一个核心公式、一个代码入口，就已经非常扎实。