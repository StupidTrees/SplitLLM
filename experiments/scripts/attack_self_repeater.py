import os
import random
import sys
from transformers import AutoTokenizer

sys.path.append(os.path.abspath('..'))
from sfl.model.gpt2.gpt2_split import GPT2SplitLMHeadModel
from rouge import Rouge
from sfl.utils import get_best_gpu
from sfl.utils import FLConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
import torch

cache_dir = '/root/autodl-tmp/sfl/models'  # 模型的缓存位置，需要修改
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
model = GPT2SplitLMHeadModel.from_pretrained("gpt2", cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = 50256

model.config_sfl(FLConfig(collect_intermediates=False, split_point_1=6, split_point_2=999), param_keeper=None)
# model = model.convert_to_lora_model(restore_top_bottom=False)
model.print_split_model()

# 恢复的评价指标选用ROUGE

save_dir = '/root/autodl-tmp/sfl/models/checkpoints'
my_rouge = Rouge()


# 测试模型的生成文本
def generate(text, md=model):
    model.train(False)
    t = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    res = md.generate(t['input_ids'].to(md.device), attention_mask=t['attention_mask'].to(md.device),
                      max_length=300, num_beams=6, no_repeat_ngram_size=2, early_stopping=True,
                      num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(res[0], skip_special_tokens=True)


def calculate_rouge(tok, outputs, batch):
    output_texts = [tok.decode(outputs.logits.argmax(dim=-1)[i], skip_special_tokens=True) for i in
                    range(len(outputs.logits))]
    hyps_and_refs = zip(output_texts, batch['input_text'])
    hyps, refs = zip(*hyps_and_refs)
    try:
        result = my_rouge.get_scores(hyps, refs, avg=True, ignore_empty=True)  # 取一个 batch 的平均
    except:
        result = {'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                  'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                  'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}
    return result


def evaluate(epc, md, tok, test_data_loader):
    md.eval()
    dl_len = len(test_data_loader)
    with torch.no_grad():
        rouge_1, rouge_2, rouge_l_f1, rouge_l_p, rouge_l_r = 0, 0, 0, 0, 0
        for step, batch in tqdm(enumerate(test_data_loader), total=dl_len):
            res = md.generate(batch['input_ids'].to(md.device), attention_mask=batch['attention_mask'].to(md.device),
                              max_length=300, num_beams=6, no_repeat_ngram_size=2, early_stopping=True,
                              num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
            result = calculate_rouge(tok, res, batch)
            rouge_1 += result['rouge-1']['f']
            rouge_2 += result['rouge-2']['f']
            rouge_l_f1 += result['rouge-l']['f']
            rouge_l_p += result['rouge-l']['p']
            rouge_l_r += result['rouge-l']['r']
    print('Epoch {} Test Rouge_1: {}, Rouge_2: {}, Rouge_l_f1: {}, Rouge_l_p: {}, Rouge_l_r: {}'.format(epc,
                                                                                                        rouge_1 / dl_len,
                                                                                                        rouge_2 / dl_len,
                                                                                                        rouge_l_f1 / dl_len,
                                                                                                        rouge_l_p / dl_len,
                                                                                                        rouge_l_r / dl_len))
    if epc % 10 == 0:
        md.save_pretrained(save_dir + '/randomized/epoch_{}_rouge_{}.pt'.format(epc, rouge_l_f1 / dl_len))
    md.train(True)
    return rouge_1 / dl_len, rouge_2 / dl_len, rouge_l_f1 / dl_len, rouge_l_p / dl_len, rouge_l_r / dl_len


def encode(examples):
    # same input and output
    input = tokenizer(examples["goal"], truncation=True, padding="max_length")
    return {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask'],
            "input_text": examples["goal"]}


def get_output(text):
    t = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    res = model(t['input_ids'].to(device), attention_mask=t['attention_mask'].to(device))
    r = tokenizer.decode(res.logits.argmax(dim=-1)[-1], skip_special_tokens=True)
    return r


dataset = load_dataset('piqa')['train']
dataset_test = load_dataset('piqa')['validation']
dataset_test = dataset_test.map(encode)
# dataset = dataset.map(encode)
# dataset.set_format(type="torch", columns=["input_ids", "input_att_mask", "output_ids", "output_att_mask", "input_text"])
dataset_test.set_format(type="torch",
                        columns=["input_ids", "attention_mask","input_text"])
dataloader = DataLoader(dataset, batch_size=6)
dataloader_test = DataLoader(dataset_test, batch_size=6)

# 开始训练Attack Model
device = get_best_gpu()
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
epoch = 100


def repeat_text(texts):
    res = []
    for text in texts:
        res.append(f"{text} Repeat:{text}")
    return res


with tqdm(total=epoch * len(dataloader)) as pbar:
    evaluate(0, model, tokenizer, dataloader_test)
    for epc in range(epoch):
        model.train(True)
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            texts = repeat_text(batch["goal"])
            batch = tokenizer(texts, truncation=True, padding="max_length", return_tensors="pt")
            input_ids = batch['input_ids'].to(device)
            output_ids = input_ids.clone().to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, labels=output_ids, attention_mask=attention_mask)
            loss = outputs.loss
            pbar.set_description(f'Epoch {epc} Loss {loss.item():.5f}')
            loss.backward()
            if (epc * len(dataloader) + step) % 2000 == 0:
                q = "To mix food coloring with sugar, you can"
                print(q, "=>", generate(q))
            optimizer.step()
            pbar.update(1)
        evaluate(epc, model, tokenizer, dataloader_test)
