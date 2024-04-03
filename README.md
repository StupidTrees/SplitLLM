## 环境准备

```shell
conda env create -n sfl python=3.11
conda activate sfl
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 模型下载
1. 前往`sfl/config`，修改dataset_cache_dir，model_download_dir，model_cache_dir为本地路径
2. 运行下载脚本，可能需要代理
```shell
cd experiments/script
python model_download.py --repo_id gpt2-large
python model_download.py --repo_id FacebookAI/roberta-large
python model_download.py --repo_id google-bert/bert-large-uncased
python model_download.py --repo_id google/flan-t5-base
python model_download.py --repo_id google/flan-ul2-base
python model_download.py --repo_id meta-llama/Llama-2-7b-chat-hf
python model_download.py --repo_id THUDM/chatglm3-6b
python model_download.py --repo_id google/flan-ul2
python model_download.py --repo_id google/vit-large-patch16-224
```


## 数据集下载
```shell
cd $dataset_cache_dir
git clone https://huggingface.co/datasets/wikitext.git
git clone https://huggingface.co/datasets/piqa.git
git clone https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K.git
git clone https://huggingface.co/datasets/knkarthick/dialogsum.git
git clone https://huggingface.co/datasets/gsm8k.git
git clone https://huggingface.co/datasets/imdb.git
git clone https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese.git
git clone https://huggingface.co/datasets/frgfm/imagewoof.git
```


## 运行实验脚本
- 运行python脚本`experiments/scripts/pipeline/[EXP]XXX.sh` 

上述过程**需要配置wandb**: 在本地环境使用

```shell
wandb login
```
来登陆自己的wandb账号，运行结果将上传至wandb可视化


## SFL模拟
训练完攻击模型吼，使用sfl_with_attacker相关的notebook/py文件/脚本，进行SFL模拟

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
