## 提示
在理解Transformer的时候应重点关注：
1. 这个过程是在“训练（Training）”还是在“推理（Inference）”
2. 公式中，“超参数（Hyperparameter）”还是“可学习参数（Trainable parameters）”

本文重点是transformer在 **推理** 时的工作原理


## 词汇
- **参数(Parameter)**
	- **超参数**: 模型设计时人为规定的
	- **可学习参数**：在训练过程中不断更新的参数，训练完后变为固定值

- **词嵌入（Word Embedding）**
	transformer的开头，只需要知道它把一个词变成一个多维的向量，在transformer中一般是512维 $a_i\in R^{512}$ ( $a_i$ 指单词)这个维度被称为“**d_{model}**”

- **训练（Training）**
	- 作用
	
	训练模型的本质就是在不断调整模型的参数
	我们把输入和输出表达为“问题”和“答案”，便于理解
	具体如何实现：
	给模型喂数据集（包括问题和答案），
	让模型先以初始的参数预测答案
	再将此预测的答案和正确答案进行比较，
	根据比较出来的差距，通过反向传播算法调整参数（这个参数影响预测的结果）
	不断重复直到此参数得到的答案接近与正确答案

	而在推理（inference）阶段，模型使用的参数就是训练（training）的时候得到的最终参数
	这个参数就是一个固定值了
	- 举例
	
	就按最简单的高中学过的线性回归模型来说：y=ax+b
	训练阶段：我们知道几组（x，y），但是不知道a，b是多少，现在我们想要用这个模型去预测一个x0对应的y是多少（此x0没有出现在已知数据集中）
	1. 假定两个a，b，输入x1，得到一个y1'
	2. 对比y1'和y1的差距（叫做 **计算损失函数loss**），调整a，b参数的大小（这个方法在训练模型的时候叫做“**反向传播算法**”）
	3. 不断循环直到输入xn之后得到的yn和正确答案几乎一样
	4. 此时得到的参数a，b就是这个模型去推理时可以使用的参数
- **推理（Inference）**
	指模型被使用的这个阶段


## Transformer的理解
前提：
1. 以将"Я видел кошку на мате <eos>” 翻译成”<bos> I saw a cat"为例，讲解这个翻译过程全流程
2. 这些过程（除了masked-self-attention板块是）都是使用模型/模型实际工作的过程（即”推理“inference），而非训练（training）
3. 按照《Attention is all you need》这篇文章中的Figure1来讲解

<img width="833" height="1143" alt="9d1187d3adb325929f174f8c19efaa72" src="https://github.com/user-attachments/assets/44428b2f-452d-4f47-874a-8b816e8befab" />

### 整体架构：
**当Transformer还未输出 `<eos>`（句子终止符）或还未达到规定的最大生成长度时（终止条件）：**
**{**
1. { "Я видел кошку на мате <eos>” }输入encoder，encoder输出
2. {（encoder的输出）+（将要预测的词的previous history） }输入Decoder，Decoder输出
3. { Decoder的输出 }输入 linear和softmax层 ，输出：下一个词的概率分布
4. 把这个词添加到这次的previous history中，作为下一次的previous history

**}**
### 分部理解：
---
#### Multi-head-Encoder-Self-attention

首先明确 attention层是 **一个词一个词**处理文本
attention的定义的对象是每一个词（比如会说“cat这个词对sat这个词的attention是...“）

Self-attention可以分为encoder-encoder attention和 decoder-decoder attention
这里就指encoder-encoder attention（即每个词的attention仅仅在input的句子里面，不会把注意力放在output上）

Multi-head+是升级版，相当于让模型不止学到一种关系模式
让每一个head都有自己的参数（ $W_i$ ）
可以从不同角度去理解句子（语义相似，句法关系，主谓关系...）

##### attention的计算
**定义**
维度 $d_k,d_v$ 是人为设定的**超参数**
而 $W^Q,W^K,W^V$ 叫做“权重矩阵”，这是通过训练得到的参数（**可学习参数**）

在transformer的最开头通过 **word embedding** 将每个词变成一个多维的向量 $a_i，a_i\in R^{d_model}$
然后规定 $d_q,d_k,d_v=d_model/h$ ( $h$ 是**头数**，一般取8，)

**步骤**
1. 对于单头attention
**初级版**
	1. Word Embedding/hidden state得到每个词的向量$a_i$
	2. 计算每个词的 $Q_i,K_i,V_i$ ：
	 $Q_i=W^Q \cdot a_i,K_i=W^K \cdot a_i,V_i=W^V \cdot a_i$,相当于把$a_i$这个$d_{model}$(512)维的向量映射到了三个$d_{model}/h$(64)维的向量[线性变换]
	3. 计算相关度$\alpha_{i,j}$（第i个词和第j个词的相关度,i和j可以相等）
	 $\alpha_{i,j}=Q_i \cdot K_j$ （64*64）
	4. 经过softmax层：(原因：1.归一化 2.都是正数。可以当作权重)
	 $\alpha_{i,j}' = \frac{\exp(\alpha_{i,j})}{\sum_{j} \exp(\alpha_{i,j})}$（分母的意思是所有得分指数的总和）（这个函数就叫做softmax函数）
	5. 更新词向量$a_i$→$b_i$：
	 $b_i=\sum_{j} \alpha_{i,j}' V_j$ （1*64）
	6. $head_k=[b_1,b_2,...,b_n]$（一句话n个词）(n*64)
	**升级版**
	就是把一个句子（假设有s个词）中所有词向量（ $1\times 512$ ）合并成一个（ $s \times 512$ ）的矩阵进行计算
	公式则变为：
	 $head_k= \text{softmax}\left( \frac{K^T Q}{\sqrt{d_k}} \right) V$

2. Multi-Head Attention

	在 Multi-Head 里，这一整套流程会被 **重复 \(h\) 次（比如 8 个头）**，  
	每个头都有自己的一组参数矩阵：
	 $$W_i^Q, \quad W_i^K, \quad W_i^V$$
	于是每个 head 得到一组不同的输出：
	 $$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$
	
	然后把所有 head 的输出拼接起来：
	 $$\text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)$$

	再线性变换回 $d_{model}$ 维：

	 $$\text{Output} = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

	WO 的作用是：    
	融合多头信息 并 保持模型维度一致性。


### Cross-attention
1. 只在encoder-decoder model中（GPT这种decoder-only的模型就不存在cross-attention）
2. Q:decoder;K,V:encoder
3. 比如在翻译任务中，decoder的“猫”在encoder的“cat”上attention很大
Q:是不是只有在翻译任务才需要cross-attention？
A:并非。从input读取信息生成output的任务都需要。

### Masked-attention

1. 只出现在decoder的self-attention中（因为encoder作为“理解”输入文本，并不需要预测下一个此）
2. 在**训练**过程中
防止模型偷看未来词语
在attention加一个右上三角矩阵M,其中右上角是-∞，其余为0
在**推理**过程中则不需要masked-self-attention，但是代码不需要改也不影响（因为模型在推理的时候本身就看不见未来的词）
 $$\text{Attention}(Q, K, V)= \underbrace{\text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}} + M\right)}_{\text{apply mask before softmax}} V$$
---

#### Feed-Forward Network
1. 作用:Attention是句子横向的连接理解，而FFN就是每个词单独更深入的理解(提取更复杂的特征)，增加模型的复杂度和表达能力。

2. 结构公式：
 $$\text{FFN}(x_i) = \text{Activation}(x_i W_1 + b_1) W_2 + b_2$$ 
 $x_i$ 是第i个token的hidden vector（维度是 $d_{model}$ ,通常512）
超参数： $Activation$ （一般是ReLU、GELU）
可学习参数：$W_1，W_2，b_1，b_2,d_{model},d_{ff}$
其中 $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ 
 $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ 
常见参数选择是 $d_{\text{ff}} = 4 \times d_{\text{model}}$
---
#### Add & Norm
##### Add: Residual Connection
1. 概念：简单来说就是把加工之前和加工之后的值相加
2. 位置：Attention、FFN之后
3. 公式： $x_{l} = \mathrm{block}(x_{l-1}) + x_{l-1}$ 这里的 $\text{block}$ 可以是 $\text{attention}$ 或者 $\text{Feedforward} $ 
4. 作用：缓解梯度消失；让梯度更容易在网络中传播，从而可以堆叠很多层 

##### Norm: Layer Normalization
1. 概念：对每个样本的特征维度进行归一化
2. 位置：Residual Connection之后
3. 公式：
 $$y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathrm{Var}[x]} + \epsilon} \, \gamma + \beta$$
其中 $\gamma,\beta$ 是可学习参数, $\epsilon$ 是超参数
4. 作用：稳定训练;加快收敛
---
##### Positional Embedding
1. 概念：位置嵌入
2. 位置：Embedding之后，Attention之前
3. 公式：
 	$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$

	 $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$
	其中 $pos$ 是词在句子中的位置， $i$ 是维度索引（ $0 \leq i \leq d_{model}-1$ ）， $d_{model}$ 是模型的隐藏层维度（一般是512）

5. 作用：让模型知道每个token的位置


