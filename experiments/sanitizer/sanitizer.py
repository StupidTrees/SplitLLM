# coding: utf-8
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'False'

from typing import Dict
import torch
import transformers
from transformers import BitsAndBytesConfig, GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import StoppingCriteriaList
import config
import spacy
from ltp import LTP
import langdetect
import numpy as np
import re
import gc
from sfl.utils.exp import *
from openai import OpenAI
import sys
sys.path.append(os.path.abspath('../../..'))

langdetect.DetectorFactory.seed = 0

client = OpenAI(
    api_key=config.API_KEY
)

tasks = ['trans']

# specify base model
base_model = "bloomz-560m"
base_model_dir = f"./models/{base_model}"

# specify langauge
lang = 'en'

# specify lora weights
sanitizer = f"./lora_weights/{base_model}_{lang}/checkpoint-6300"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def smart_tokenizer_and_embedding_resize(
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """
    Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    special_tokens_dict: Dict[str, str] = {}
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = config.DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict['eos_token'] = config.DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = config.DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict['unk_token'] = config.DEFAULT_UNK_TOKEN
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def merge_spans(intervals):
    """
    Merge overlapping interval.
    """
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged


def merge_labeled_spans(spacy_list, text, return_positions=False):
    """
    Merge overlapping intervals according to labels.
    """
    merged_list = []
    for s, e, label in spacy_list:
        if merged_list and merged_list[-1][1] == s and merged_list[-1][2] == label:
            merged_list[-1][1] = e
        else:
            merged_list.append([s, e, label])
    if return_positions:
        return merged_list
    merged_list = {text[s:e] for s, e, _ in merged_list}
    return merged_list


def get_merged_spans(text, ents):
    """
    Obtain the position intervals of entities and merge overlapping parts simultaneously.
    """
    all_spans = []
    for ent in ents:
        try:
            spans = [[match.start(), match.end()] for match in re.finditer(ent, text)]
            all_spans.extend(spans)
        except:
            pass
    merged_spans = np.array(merge_spans(all_spans))
    return merged_spans


def get_ents_zh(text, ltp, spacy_model):
    """
    Entity-level.
    Note: Return a deduplicated list of entities, convert to string when necessary.
    """
    label_set = config.MID_ENTITY_SET
    ner_list = {ent for _, ent in ltp.pipeline([text], tasks=["cws", "pos", "ner"]).ner[0]}
    spacy_list = [(ent.start_char, ent.end_char, ent.label_) for ent in spacy_model(text).ents if
                  ent.label_ in label_set]
    spacy_list = merge_labeled_spans(spacy_list, text)
    work_of_art = set(re.findall(r'(《.*?》)', text) + re.findall(r'(“.*?”)', text))
    ner_list = list(set.union(ner_list, spacy_list, work_of_art))
    return ner_list


def get_low_ents_en(text, spacy_model):
    """
    Entity-level (low-entity).
    Note: Only `PERSON` type entities are extracted.
    """
    label_set = config.LOW_ENTITY_SET
    spacy_list = [(ent.start_char, ent.end_char, ent.label_) for ent in spacy_model(text).ents if
                  ent.label_ in label_set]
    spacy_list = merge_labeled_spans(spacy_list, text)
    ner_list = list(spacy_list)
    return ner_list


def get_ents_en(text, spacy_model):
    """
    Entity-level.
    Note: Return a deduplicated list of entities, convert to string when necessary.
    """
    label_set = config.HIGH_ENTITY_SET
    spacy_list = [(ent.start_char, ent.end_char, ent.label_) for ent in spacy_model(text).ents if
                  ent.label_ in label_set]
    spacy_list = merge_labeled_spans(spacy_list, text)
    ner_list = list(spacy_list)
    return ner_list


def get_entity_labelled_text(text, spacy_model, return_ents=True):
    """
    Entity-level.
    Note: Output text with anonymized labels, optionally return the corresponding list of entities.
    """
    label_set = config.HIGH_ENTITY_SET
    doc = spacy_model(text)
    spacy_list = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents if
                  ent.label_ in label_set]
    spacy_list = merge_labeled_spans(spacy_list, text, return_positions=True)
    positions = np.array([ent[:2] for ent in spacy_list])
    labels = [ent[2] for ent in spacy_list]
    ner_list = set()
    for i in range(len(spacy_list)):
        s, e = positions[i]
        label = labels[i]
        ner_list.add(text[s:e])
        text = text[:s] + f'<{label}>' + text[e:]
        positions[i:, :] = positions[i:, :] + len(label) + 2 - (e - s)
    if return_ents:
        return text, list(ner_list)
    else:
        return text


def get_contextual_labelled_text(text, spacy_model, return_ents=True, context_level="LOW"):
    """
    Contextual-level.

    Args:
        context_level: (`LOW` or `HIGH`)
            Different privacy levels.
    """
    doc = spacy_model(text)
    spacy_list = []
    if context_level == "LOW":
        for token in doc:
            if "subj" in token.dep_.lower():
                spacy_list.append((token.idx, token.idx + len(token.text), "SUBJ"))
            elif "obj" in token.dep_.lower():
                spacy_list.append((token.idx, token.idx + len(token.text), "OBJ"))
            elif token.dep_ == "ROOT":
                spacy_list.append((token.idx, token.idx + len(token.text), "ROOT"))
    elif context_level == "HIGH":
        for token in doc:
            if token.pos_ in ["PROPN", "PRON", "VERB"]:
                spacy_list.append((token.idx, token.idx + len(token.text), token.pos_))

    spacy_list = merge_labeled_spans(spacy_list, text, return_positions=True)
    positions = np.array([ent[:2] for ent in spacy_list])
    labels = [ent[2] for ent in spacy_list]
    ctx_list = set()
    for i in range(len(spacy_list)):
        s, e = positions[i]
        label = labels[i]
        ctx_list.add(text[s:e])
        text = text[:s] + f'<{label}>' + text[e:]
        positions[i:, :] = positions[i:, :] + len(label) + 2 - (e - s)
    if return_ents:
        return text, list(ctx_list)
    else:
        return text


def get_labelled_text_with_id(text, spacy_model, return_ents=True):
    """
    Entity-level.
    Label sanitization function with appended ID.
    Note: Output text with anonymized labels, optionally return the corresponding list of entities.
    """
    label_set = config.HIGH_ENTITY_SET
    label_set = {k: {'<cur_id>': 0} for k in label_set}
    spacy_list = [(ent.start_char, ent.end_char, ent.label_) for ent in spacy_model(text).ents if
                  ent.label_ in label_set.keys()]
    spacy_list = merge_labeled_spans(spacy_list, text, return_positions=True)
    positions = np.array([ent[:2] for ent in spacy_list])
    labels = [ent[2] for ent in spacy_list]
    ner_list = set()
    for i in range(len(spacy_list)):
        s, e = positions[i]
        ent_text = text[s:e]
        label = labels[i]
        if ent_text not in label_set[label]:
            label_set[label][ent_text] = label_set[label]['<cur_id>']
            label_set[label]['<cur_id>'] += 1
        label = f'<{label}_{label_set[label][ent_text]}>'
        ner_list.add(ent_text)
        text = text[:s] + label + text[e:]
        positions[i:, :] = positions[i:, :] + len(label) - (e - s)
    if return_ents:
        return text, list(ner_list)
    else:
        return text


def mark_ents(text, spacy_model, return_ents=True):
    """
    Identify entities in the text and enclose them with '<>'.
    """
    label_set = config.HIGH_ENTITY_SET
    spacy_list = [(ent.start_char, ent.end_char, ent.label_) for ent in spacy_model(text).ents if
                  ent.label_ in label_set]
    spacy_list = merge_labeled_spans(spacy_list, text, return_positions=True)
    positions = np.array([ent[:2] for ent in spacy_list])
    ner_list = set()
    for i in range(len(spacy_list)):
        s, e = positions[i]
        ner_list.add(text[s:e])
        text = text[:s] + f'<{text[s:e]}>' + text[e:]
        positions[i:, :] = positions[i:, :] + 2
    if return_ents:
        return text, list(ner_list)
    else:
        return text


def sanitize_text(raw_input, target_ents, model, tokenizer, ltp, spacy_model, lang):
    """
    Integrated sanitization function.

    Args:
        target_ents: (`label` or `auto`)
            Different sanitization strategies:
                - `label`: use spacy.
                - `auto`: use fine-tuned bloomz model.
    """
    sub_model = PeftModel.from_pretrained(
        model, sanitizer, quantization_config=bnb_config, device_map='cuda:0', trust_remote_code=True
    )

    if target_ents == 'label':
        # can be other label sanitization functions defined above
        # entity-level
        # return get_entity_labelled_text(raw_input, spacy_model, return_ents=False)
        # contextual-level (low)
        # return get_contextual_labelled_text(raw_input, spacy_model, return_ents=False, context_level="LOW")
        # contextual-level (high)
        return get_contextual_labelled_text(raw_input, spacy_model, return_ents=False, context_level="HIGH")

    # use automatic sanitization model
    if target_ents == 'auto':
        if lang == 'en':
            target_ents = get_ents_en(raw_input, spacy_model)
        else:
            target_ents = get_ents_zh(raw_input, ltp, spacy_model)
        print(target_ents)

    with open(f'./prompts/sanitize_{lang}.txt', 'r', encoding='utf-8') as f:
        initial_prompt = f.read()
    input_text = initial_prompt % (raw_input, target_ents)
    input_text += tokenizer.bos_token
    inputs = tokenizer(input_text, return_tensors='pt')
    inputs = inputs.to('cuda:0')
    len_prompt = len(inputs['input_ids'][0])

    def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        cur_top1 = tokenizer.decode(input_ids[0, len_prompt:])
        if '\n' in cur_top1 or tokenizer.eos_token in cur_top1:
            return True
        return False

    pred = sub_model.generate(
        **inputs,
        generation_config=GenerationConfig(
            max_new_tokens=int(len(inputs['input_ids'][0]) * 1.3),
            do_sample=False,
            num_beams=3,
            repetition_penalty=5.0,
        ),
        stopping_criteria=StoppingCriteriaList([custom_stopping_criteria])
    )
    pred = pred.cpu()[0][len(inputs['input_ids'][0]):]
    response = tokenizer.decode(pred, skip_special_tokens=True).split('\n')[0]
    torch.cuda.empty_cache()
    gc.collect()
    return response


if __name__ == '__main__':
    # load model
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, load_in_4bit=True, quantization_config=bnb_config,
                                                 device_map='cuda:0', trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    # model, tokenizer = get_model_and_tokenizer("bloomz")
    smart_tokenizer_and_embedding_resize(tokenizer=tokenizer, model=model)
    spacy_model = spacy.load(f'{lang}_core_web_trf')
    # only chinese uses ltp
    ltp = LTP()

    # demo
    raw_text = "November 7, 2023 (Washington, DC) — A massive surge of corruption and organized crime in Libya orchestrated by the country’s rulers threatens the survival of essential institutions, is generating cascading regional and international security risks, and could spark major armed conflict, warns a new report released today by The Sentry."

    label_sanitize_text = sanitize_text(raw_text, "label", model, tokenizer, ltp, spacy_model, lang)
    print('\033[1;94mSanitized Text (label):\033[0m ', label_sanitize_text)
    auto_sanitize_text = sanitize_text(raw_text, "auto", model, tokenizer, ltp, spacy_model, lang)
    print('\033[1;32mSanitized Text (auto):\033[0m ', auto_sanitize_text)
