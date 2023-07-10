# 工作清单



# 2023/7/10 

## 主要事项
  - （余总）讲话
  -  上周工作总结
  -  本周安排

## 上周任务完成情况
  - 读文档和论文，了解量化及相关研究（图融合）
  

## 当前问题汇总
 - 版本、工具库（cmake, gcc ，ninja ）： 后续由盐城方面协助统一解决
 - 安排近期见面交流  

## 本周工作内容
  -  （刘  涛）Training Transformers with 4-bit Integers （https://arxiv.org/abs/2306.11987, 朱军）
  -  （朱梦静）继续复现FP8训练（CV, NLP, Transformer）
     




# 2023/7/3 

## 主要事项
 - 明确工作方向:以研发为主，产研结合
 - 熟悉工作环境


## 研究方向
 - 神经网络和量化
 - 高性能计算(HPC)
 - Security


## 神经网络和量化
### 阅读文档和论文
   - NVIDIA H100 Tensor Core GPU Architecture(White paper)
   - FP8 Formats for Deep Learning， arXiv:2209.05433
   - FP8 Quantization: The Power of the Exponent， arXiv:2208.09225
   - 
### 工作内容
   - 入门： 下载数据集，训练resnet18, BERT模型     
   - 量化： 复现在训练中使用FP8量化的结果； 要求能读懂论文和开源代码
   - 可选任务： 收集目前已开源的大语言模型，如LLaMA，完成一次微调
 
## Security
>### 加密技术
 > > - 多方计算(MPC)
> > 
 > > - 同态加密(HE):    [同态加密开源库、工具、框架](https://github.com/jonaschn/awesome-he)
>### 可信执行环境(TEE)
> >  - TDX
> >  - SGX
> >  - TrustZone
> >  - SVE
> >  - Penglai(蓬莱)
> >  - H100 MIG-level TEE
     
## 高性能计算HPC
>  - 冷冻电镜


## 开发环境
 **注意事项： 不要在开发主机上运行与工作无关的软件**


- 使用步骤：先接入VPN，然后使用xshell等终端软件登录超算集群

- 需要先下载客户端，然后填入账号密码，即可登录。
- VPN 入口： https://vpn.meta-stone.cn:6443 
-   VPN账号： 
-   VPN密码：



- 登录开发主机节点：
- 主机地址： 
- 账号密码：  

## 完成情况

## 问题汇总


