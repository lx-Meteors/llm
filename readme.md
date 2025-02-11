# 一、预训练
## 数据集
wiki：https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered?row=15
## 遇到问题
不同层级下的导包问题
```
import sys
sys.path.append('/mnt/zhaorunsong/lx/My-LLM')
```
## 基本流程
1. 定义各种路径，超参数
2. 加载数据集 -> 处理数据集
3. 将超参数放在训练器中初始化
4. 开始训练

# 二、微调
## 数据集
alpaca：https://huggingface.co/datasets/shibing624/alpaca-zh
## 遇到的问题
将数据处理成需要的格式 | 数据长度统一 | 只给答案作标签
# 三、DPO
## 数据集
还是sft阶段的alpaca，不过需要将sft输出的答案设置为reject、正常的数据集答案是 chosen
prompt：指令
chosen：正常数据集答案
reject：sft生成的答案
## 处理数据集