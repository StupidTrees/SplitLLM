## 环境准备

```shell
conda env create -n sfl python=3.11
conda activate sfl

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 实现说明

模拟SFL的实现，即一个模型分为

| Bottom Layers | Trunk Layers（Adapters） | Top Layers |
|---------------|------------------------|------------|

其中Trunk是一个模型的主干部分，Bottom-Layers和Top-Layers是模型的头尾部分。

为了进行SFL模拟，**不采用在代码实现上对模型进行真实分割的方式，而采用对模型的不同部分参数进行独立维护的方式**，且以串行方式模拟Client训练，并不进行真实的Client并行。

模拟过程如下：

1. 某一轮联邦学习开始，选取Client 0, 1, 2
2. 从磁盘上加载Client 0的模型参数（包含bottom和top layers，以及它对应的一份Server上的trunk）到GPU上的模型中
3. Client 0进行本地训练，更新模型所有参数，过程和集中式学习一致
4. Client 0训练完成，保存模型参数到磁盘
5. 从磁盘上加载Client 1的模型参数到GPU上的模型中
6. ...
7. 该轮联邦学习结束，聚合磁盘上的所有client对应的那份trunk参数得到平均trunk，然后把所有client的trunk参数更新为平均trunk
8. 重复1-7

## Example

使用GPT2作为Example。GPT2有12个Transformer Blocks，以Block为单位进行切分，

| Bottom Layers | Trunk Layers（Adapters） | Top Layers |
|---------------|------------------------|------------|
| bottom parts + 1 Blocks| 10 Blocks|  1 Blocks + top parts|
