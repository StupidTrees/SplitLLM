
<p align="left"><img src="./doc/img/logo.png" width=400></p>

# SplitLLM: Split Learning Framework for Privacy Attacks
SplitLLM is a Split Learning simulation framework designed for Large Language Models (LLMs).
It enables flexible and scalable model fine-tuning under a split learning architecture. 
The framework is compatible with Hugging Face models.

SplitLLM supports extensible integration of privacy attack experiments, including mainstream DRAs like DLG, TAG, LAMP. 
The proposed Bidirectional Semi-white-box Reconstruction (BiSR) attack is also demonstrated in the example.

### Quick Start

### Environment Setup

```shell
conda env create -n sfl python=3.11
conda activate sfl
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Model Download
1. Go to `sfl/config` and modify `dataset_cache_dir`，`model_download_dir`，`model_cache_dir` to your own path
2. Run the following commands to download models, some of them may require a qualified huggingface token
```shell
cd experiments/script
python model_download.py --repo_id meta-llama/Llama-2-7b-chat-hf
python model_download.py --repo_id gpt2-large
python model_download.py --repo_id THUDM/chatglm3-6b
...
#python model_download.py --repo_id FacebookAI/roberta-large
#python model_download.py --repo_id google-bert/bert-large-uncased
#python model_download.py --repo_id google/flan-t5-base
#python model_download.py --repo_id google/flan-ul2-base
#python model_download.py --repo_id meta-llama/Meta-Llama-3-8B 
#python model_download.py --repo_id lucyknada/microsoft_WizardLM-2-7B
#python model_download.py --repo_id lmsys/vicuna-7b-v1.5
#python model_download.py --repo_id tiiuae/falcon-7b-instruct
#python model_download.py --repo_id Salesforce/codegen25-7b-instruct_P
#python model_download.py --repo_id EleutherAI/gpt-j-6b
#python model_download.py --repo_id google/flan-ul2
#python model_download.py --repo_id google/vit-large-patch16-224
```

### Download Datasets
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


### Run the Experiment Demo
- Run the script in `experiments/scripts/pipeline/demo_bisr.sh` 

Note that the script requires wandb to be installed and configured.

## Split Learning Simulation

In SL, a model is divided into three parts:

| Bottom Layers | Trunk Layers（Adapters） | Top Layers |
|---------------|------------------------|------------|

where Bottom-Layers and Top-Layers are input and output end of the model, and Trunk-Layers are the middle part of the model. 


To simulate Split Federated Learning (SFL), we do not employ the approach of physically splitting the model in code implementation. Instead, we independently maintain different parts of the model's parameters. We simulate Client training in a serial manner, without actual Client parallelism.

The simulation process is as follows:

1. At the start of a round of federated learning, select Clients 0, 1, and 2.
2. Load the model parameters of Client 0 (including bottom and top layers, and its corresponding trunk on the Server) from disk into the GPU model.
3. Client 0 performs local training, updating all model parameters, following the process consistent with centralized learning.
4. Once Client 0 completes training, save the model parameters to disk.
5. Load the model parameters of Client 1 from disk into the GPU model.
6. ...
7. At the end of the federated learning round, aggregate the trunk parameters corresponding to all clients on disk to obtain the average trunk. Then, update the trunk parameters of all clients to the average trunk.
8. Repeat steps 1-7. 