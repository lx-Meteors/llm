# Pretrain Tips Digest

本文件整理 `models/model_meteor.py` 中的 Tips/TODO，升级为“问题 -> 直觉 -> 公式 -> shape -> 最小例子 -> 检查点”。

---

## 1) RoPE 频率为什么预计算

- 问题：为什么 `precompute_freqs_cis` 要提前计算 cos/sin。
- 直觉：每层每个 token 都会重复使用同一组位置角度，提前算好等于把“查表”代替“重复算三角函数”。
- 核心公式：

$$
\omega_i = \theta^{-\frac{2i}{d}},\quad
\phi_{t,i}=t\cdot \omega_i
$$

其中 $t$ 是位置，$i$ 是通道对索引，$d$ 是 head 维度。
- shape：
  - `freqs`: `[T, d/2]`
  - `freqs_cos/freqs_sin` 拼接后常用为 `[T, d]`
- 最小例子：`T=512, d=64`，预计算一次得到 `[512,64]` 的 cos/sin，所有层共享。
- 检查点：不同 batch 重复调用前向时，不应重复构建完整 cos/sin 张量。

## 2) 为什么 RoPE 是“旋转”

- 问题：为什么要在 q/k 上做二维子空间旋转。
- 直觉：把每两维看成 2D 平面向量，位置越靠后，旋转角度越大。
- 核心公式：

$$
R(\phi)=
\begin{bmatrix}
\cos\phi & -\sin\phi\\
\sin\phi & \cos\phi
\end{bmatrix}
$$

对 query/key 应用后，点积里会自然出现“相对位移”信息。
- 最小例子：若某一对通道是 $(a,b)$，旋转后变为 $(a\cos\phi-b\sin\phi,\ a\sin\phi+b\cos\phi)$。
- 检查点：只旋转 q/k，不旋转 v。

## 3) `transpose` 和 `repeat_kv` 的目的

- 问题：`xq/xk/xv` 为什么要转置，`repeat_kv` 在做什么。
- 直觉：注意力核要求 head 维在前；而 GQA 下 q 头多、kv 头少，要把 kv 复制到同数量级才能做并行矩阵乘。
- shape 追踪：
  - 线性层后：`xq=[B,T,Hq,D]`, `xk/xv=[B,T,Hk,D]`
  - `repeat_kv` 后：`xk/xv=[B,T,Hq,D]`
  - `transpose(1,2)` 后：`[B,Hq,T,D]`
- 最小例子：`Hq=8, Hk=2`，`n_rep=4`，每个 kv 头复制 4 次。
- 检查点：需保证 `Hq % Hk == 0`。

## 4) 为什么要 `xk.transpose(-2, -1)`

- 问题：点积注意力里 key 为什么交换最后两个维度。
- 直觉：这是标准矩阵乘法规则，不转置就内积不上。
- 核心公式：

$$
  ext{scores}=\frac{QK^\top}{\sqrt{d_k}}
$$

- shape：`Q=[B,H,T_q,D]`, `K^T=[B,H,D,T_k]`，结果 `scores=[B,H,T_q,T_k]`。
- 检查点：`scores` 最后一维应等于 `T_k`。

## 5) 为什么要 causal mask

- 问题：`scores + mask` 的意义。
- 直觉：第 $t$ 个 token 只能看见 $\le t$ 的历史，不能作弊看未来。
- 核心公式：

$$
M_{ij}=
\begin{cases}
0,& j\le i\\
-\infty,& j>i
\end{cases}
$$

`softmax(scores + M)` 会把未来位置概率压到 0。
- 最小例子：长度 4 的 mask 是上三角（不含对角）全 `-inf`。
- 检查点：推理阶段若有 cache，也要保证新 token 只 attending 到历史。

## 6) 为什么要 softmax

- 问题：注意力分数为什么必须 softmax。
- 直觉：注意力本质是“按权重加权平均”，权重需要可比较、可归一。
- 核心公式：

$$
\alpha_i=\frac{e^{s_i}}{\sum_j e^{s_j}},\quad \sum_i \alpha_i=1
$$

- 补充：除以 $\sqrt{d_k}$ 是为防止分数方差随维度增大而过大，避免 softmax 饱和。
- 检查点：实现中常先转 `float32` 做 softmax，再 cast 回原 dtype。

## 7) 注意力后和 FFN 后为什么都 dropout

- 问题：看起来像“重复 dropout”。
- 直觉：两个子层承担不同功能，分别正则化更稳。
- 结构视角：
  - `Attention -> proj -> dropout -> residual`
  - `MLP -> dropout -> residual`
- 检查点：推理 `eval()` 下 dropout 自动关闭。

## 8) `intermediate_size` 为什么按 64 对齐

- 问题：`64 * ceil(raw/64)` 的原因。
- 直觉：硬件更偏好固定块大小，维度对齐通常带来更好的 kernel 利用率。
- 例子：`raw=1365`，对齐后是 `1408`。
- 检查点：对齐后参数量会略增，需在吞吐和显存间平衡。

## 9) GLU 变体为什么有效

- 问题：`act(gate_proj(x)) * up_proj(x)` 的作用。
- 直觉：一条分支做“内容”，一条分支做“门控”，乘法融合后表达更细。
- 核心公式（SwiGLU 形式）：

$$
  ext{FFN}(x)=W_d\big(\text{silu}(W_gx)\odot (W_ux)\big)
$$

- 检查点：门控分支和内容分支维度必须一致。

## 10) 为什么先 Attention 再 MLP 且每层都 Norm

- 问题：Transformer block 顺序与 norm 必要性。
- 直觉：先“看别人”（attention），再“加工自己”（MLP）。
- Pre-Norm 形式：

$$
x \leftarrow x + \text{Attn}(\text{Norm}(x)),\quad
x \leftarrow x + \text{MLP}(\text{Norm}(x))
$$

- 实践意义：深层模型里 Pre-Norm 通常更容易训稳，梯度更顺。
- 检查点：观察深层训练时 loss 是否抖动或梯度爆炸。

## 11) `ModuleList` 与 `register_buffer`

- 问题：为什么用 `ModuleList`，为什么注册 buffer。
- 直觉：
  - `ModuleList` 让优化器“看见”每一层参数。
  - buffer 让状态“跟着模型走”，但不会被更新。
- 参数 vs buffer：
  - parameter：出现在 `model.parameters()`，参与梯度更新。
  - buffer：出现在 `state_dict()`（若 persistent=True），不参与反向传播。
- 检查点：RoPE cos/sin 用 buffer 是合理选择，不应进入 optimizer。

## 12) TODO 状态

- MoE 分支尚未在当前训练入口启用完整实现。
- `aux_loss` 相关逻辑还需要与 MoE 前向输出联调。

## 13) 建议补全顺序

1. 先补 shape 注释：attention 前后张量维度。
2. 再补数学注释：RoPE、softmax、causal mask 三块。
3. 最后补实验对照：
   - 对齐维度吞吐对比
   - 有/无 dropout 对比
   - Pre-Norm 稳定性对比
