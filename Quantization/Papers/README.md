# 4-bit and 8-bit Papers 整理

## 1 - [Logarithmic unbiased quantization: Practical 4-bit training in deep learning](https://arxiv.org/pdf/2112.10769v1.pdf)

链接：https://arxiv.org/abs/2112.10769v1

- 研究了无偏舍入对前向和后向的影响。表明，rounding-to-nearest(RDN)适用于前向阶段，而stochastic rounding(SR)是更适合于后向阶段。虽然SR是无偏的，它相比于RDN一般有更差的均方误差。

- 使用LUQ(其中包括SMP: REDUCING THE VARIANCE WHILE KEEPING IT UNBIASED和FNT: FINE-TUNING IN HIGH PRECISION FOR ONE EPOCH)将梯度量化为FP4,将权重和激活量化到INT4。

## 2 - [Ultra-low precision 4-bit training of deep neural networks](https://papers.nips.cc/paper/2020/file/13b919438259814cd5be8cb45877d577-Paper.pdf)

链接：https://dl.acm.org/doi/10.5555/3495724.3495876

- 一种新的每层可训练梯度缩放技术——GradScale，将梯度与FP4范围对齐，基本上最大限度地利用了每层的范围，并规避了在所有层中使用全局损失缩放因子（APEX）的问题。同时提出一种双阶段量化技术(Two-Phase Rounding ,TPR)，可最小化FP4梯度的均方和预期误差的量化误差。

- 将梯度量化为FP4,将权重和激活量化到INT4。特殊的，比如更深入的模型（ResNet 50和ResNet 101），利用FP8计算Conv1x1层；在MobileNet-V2中，将FP4应用于后向GEMM，同时将FP8梯度用于更新GEMM。

## 3 - [Training high-performance and large-scale deep neural networks with full 8-bit integers](https://arxiv.org/pdf/1909.02384.pdf)

链接：https://arxiv.org/abs/1909.02384

## 4 - [Towards unified int8 training for convolutional neural network](https://arxiv.org/pdf/1912.12607.pdf)

链接：https://arxiv.org/abs/1912.12607

## 5 - [A Statistical Framework for Low-bitwidth Training of Deep Neural Networks](https://arxiv.org/pdf/2010.14298.pdf)

链接：https://arxiv.org/abs/2010.14298

- Per-Sample Quantizer(PSQ)处理样本动态范围变化大

- **Block Householder Quantizer (BHQ)**量化 梯度

梯度中只有少量的行是重要的，将行分为多个group，再应用Householder变换得到一个Householder矩阵和一个对角矩阵,降低运算复杂度

也是利用梯度矩阵稀疏性的处理，但不是很理解

## 6 - [FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer](https://arxiv.org/pdf/2111.13824.pdf)

链接：https://arxiv.org/abs/2111.13824

- 使用Power-of-Two Factor(PTF)对ViT中的 LayerNorm 量化，不同通道不同Factor

- 对 Attention Map使用 log2量化，利用Attention map的稀疏性量化为int8/4两种不同位宽 （类似朱军论文中FP的Hadamard Quantization）

这篇看完感觉不是严格意义上的FQT，他是把Transformer中所有模块做了量化然后做了PTQ

## 7 - [Training Transformers with 4-bit Integers](http://export.arxiv.org/pdf/2306.11987)

链接：https://arxiv.org/abs/2306.11987

## 8 - [Distribution Adaptive INT8 Quantization for Training CNNs](https://arxiv.org/pdf/2102.04782.pdf)

链接：https://arxiv.org/abs/2102.04782

## 最终整理
| Paper Code | CNN | Transformer+WMT | BERT | Mobilenet |   forward   |   backward     |       Mobilenet error        |
|:----------:|:---:|:---------------:|:----:|:---------:|:-----------:|:--------------:|:----------------------------:|
| 1          |  √  |        √        |  ×   |     √     |  Int4       |  FP4 [1,3,0]   |        ImageNet -1.77%       |
| 2          |  √  |        √        |  ×   |     √     |  Int4       |  FP4 [1,3,0]   | cifar10 -0.61%,ImageNet -2.2%|
| 3          |  √  |        ×        |  ×   |     ×     |  Int8       |  Int8          |                              |
| 4          |  √  |        ×        |  ×   |     √     |  Int8       |  Int8          |cifar10 -1.01%,ImageNet -1.19%|
| 5          |  √  |    ×WMT,√IWSL   |  ×   |     ×     |  Int8/Int5  |  Int8/Int5     |                              |
| 6          |  √  |    +imagenet    |  ×   |     ×     |  Int8/Int4  |  Int8          |                              |
| 7          |  ×  |        √        |  √   |     ×     |  Int4(LSQ)  |  Int4(BS+LSS)  |                              |
| 8          |  √  |        ×        |  ×   |     √     |  Int8       |  Int8          |cifar10 -0.36%,ImageNet -0.52%|
