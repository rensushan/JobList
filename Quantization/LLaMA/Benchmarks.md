# LLM Benchmarks

 

## Open LLM Leaderboard

`RL` = RL-tuned model

`PT` = pretrained model

|      | model                                                        | Average | ARC   | HellaSwag | MMLU  | TruthfulQA |
| ---- | ------------------------------------------------------------ | ------- | ----- | --------- | ----- | ---------- |
| `PT` | [ meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf) | 67.35   | 67.32 | 87.33     | 69.83 | 44.92      |
| `RL` | [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) | 66.8    | 64.59 | 85.88     | 63.91 | 52.8       |
| `RL` | [ meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | 59.93   | 59.04 | 81.94     | 54.64 | 44.12      |
| `PT` | [ meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf) | 58.66   | 59.39 | 82.13     | 55.77 | 37.38      |
|      |                                                              |         |       |           |       |            |
| `RL` | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | 56.34   | 52.9  | 78.55     | 48.32 | 45.57      |
| `PT` | [ meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | 54.32   | 53.07 | 78.59     | 46.87 | 38.67      |
|      |                                                              |         |       |           |       |            |



## ARC

[[1803.05457\] Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge (arxiv.org)](https://arxiv.org/abs/1803.05457)



​	<u>AI2 Reasoning Challenge</u>. **多选式阅读理解**。

​	**数据划分**：划分为**一个挑战集**(2590个)和**一个简单集**(5197个)，其中挑战集只包含 被基于检索的算法和单词共现算法 **回答错误的问题**(questions answered incorrectly by both a retrieval-based algorithm and a word co-occurence algorithm)。该数据集仅包含自然的小学科学问题，并且是此类问题中最大的公共领域集(7,787个问题)。

​	**ARC语料库**：提供与该任务相关的包含14M科学事实。(Use of the Corpus for the Challenge is optional).



|       | Challenge | Easy | Total |
| ----- | --------- | ---- | ----- |
| Train | 1119      | 2251 | 3370  |
| Dev   | 299       | 570  | 869   |
| Test  | 1172      | 2376 | 3548  |

### 评价指标

单个问题：选出正确的 +1分，选出K个且包含正确的 +$\frac{1}{k}$分

总体评价：
$$
Percent = \frac{\sum_{m=1}^{M} point}{M} * 100
$$

## HellaSwag

[[1905.07830\] HellaSwag: Can a Machine Really Finish Your Sentence? (arxiv.org)](https://arxiv.org/abs/1905.07830)

**常识推理**任务。



​	数据集使用了对抗过滤(Adversarial Filtering, AF)技术生成。生成器为GPT，判别器为BERT，主打**增加上下文的多样性**和**文本长度**。

<img src="https://github.com/rensushan/JobList/blob/main/Quantization/LLaMA/pics/AF.png" style="zoom: 80%;" />

​	不断迭代，不断生成新的文本去替换易辨别的文本，直到趋于稳定。



​	作者通过**删除上下文**和**打乱ending中词语**的方式证明了**BERT类语言模型在fine-tune过程中是在每个(context, answer)对上进行词汇推理**。(systems primarily learn to detect distributional stylistic patterns during finetuning)

<img src="[https://github.com/rensushan/JobList/blob/main/Quantization/LLaMA/pics/AF.png](https://github.com/rensushan/JobList/blob/main/Quantization/LLaMA/pics/AF%20iterate.png)" />

​	SWAG中生成句子和人类编写的endings差距很大，利用深层语言模型作为AF中的生成器可以有效降低BERT判别器的精度。

### SWAG到HellaSwag的迁移

<img src="[https://github.com/rensushan/JobList/blob/main/Quantization/LLaMA/pics/AF.png](https://github.com/rensushan/JobList/blob/main/Quantization/LLaMA/pics/swag_transfer_hellaswag.png)" />

在SWAG数据集不包含的wikiHow领域，迁移学习的效果表现差，这表明，SWAG的指标是不足以证明学习通用常识推理的能力。



## MMLU

[[2009.03300\] Measuring Massive Multitask Language Understanding (arxiv.org)](https://arxiv.org/abs/2009.03300)



MMLU是一个包含**57个多选问答任务的英文评测数据集**，涵盖了初等数学、美国历史、计算机科学、法律等，难度覆盖高中水平到专家水平的人类知识。由于LLaMA中书籍与学术论文的数据组成较少，相对于Gopher、Chinchilla等模型在MMLU上的表现略逊一筹。



由于HellaSwag数据集的精度已经接近人类水平，所以提出了难度更高的MMLU。

|             | few-shot development set | validation set | test set |
| ----------- | ------------------------ | -------------- | -------- |
| per subject | 5                        | -              | 100+     |
| total       | 289                      | 1540           | 14079    |
|             |                          |                |          |

## TruthfulQA

[[2109.07958\] TruthfulQA: Measuring How Models Mimic Human Falsehoods (arxiv.org)](https://arxiv.org/abs/2109.07958)



**针对性的评估LLM的真实性**。



### 为什么会输出虚假信息？

1.模型训练没有有效泛化。           ==>   增大模型参数

2.训练数据中可能存在虚假信息。==>   本数据集评估的**主要目标**，学的越多越容易泛化到虚假信息。



**标准**：模型的**输出只包含真理/完全真实正确(事实性)的内容时**，才被认为**是正确的**

​           模型输出**“我不知道/无可奉告”等完全无信息量的回答**是被认为**是真实的**  ==>  增加了**信息性(informative)**的评价指标。



### 评价方式

使用经过微调的GPT-judge预测人类对真实性的评价，准确率在90-96%。人类的准确率为89.5%。

GPT-judge的微调方法：

1. 构建三元组训练数据(question, answer, label)label表示该answer是否正确。对于信息性来说label就是包含信息量的分数（貌似是informative就为1，uninformative就为0）。
2.  输入question和answer，loss应该就是cross entropy。

### 总结

在TruthfulQA上表现好，不代表特定领域一定真实。

但在TruthfulQA上表现不好，一定是鲁棒性差的表现。
