# 梯度劫持攻击
import dataclasses
import os

import torch
from tqdm import tqdm
from transformers import PretrainedConfig, PreTrainedModel

from sfl.config import DRA_train_label, fsha_path, DRA_test_label
from sfl.model.attacker.sip_attacker import GRUDRInverter, LSTMDRAttackerConfig
from sfl.utils.model import calc_unshift_loss, evaluate_attacker_rouge


def get_ae_save_path(md, save_dir, args):
    model_name = args.model_name
    if model_name == 'llama2':
        model_name += f'-{args.load_bits}bits'
    if ',' in args.dataset:
        p = os.path.join(save_dir,
                         f'{model_name}/{args.dataset}/Tr{args.dataset_train_frac:.3f}-Ts*{args.dataset_test_frac:.3f}'
                         f'/{md.fl_config.attack_mode}-{md.fl_config.split_point_1 if md.fl_config.attack_mode == "b2tr" else md.fl_config.split_point_2}/')
    else:
        p = os.path.join(save_dir,
                         f'{model_name}/{args.dataset}/{DRA_train_label[args.dataset]}*{args.dataset_train_frac:.3f}-{DRA_test_label[args.dataset]}*{args.dataset_test_frac:.3f}'
                         f'/{md.fl_config.attack_mode}-{md.fl_config.split_point_1 if md.fl_config.attack_mode == "b2tr" else md.fl_config.split_point_2}/')
    return p


def evaluate_ae(epc, llm, attacker, tok, test_data_loader, args):
    """
    恢复的评价指标选用ROUGE
    :return: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-L-P, ROUGE-L-R
    """
    attacker.eval()
    dl_len = len(test_data_loader)
    with torch.no_grad():
        rouge_1, rouge_2, rouge_l_f1, rouge_l_p, rouge_l_r = 0, 0, 0, 0, 0
        for step, batch in tqdm(enumerate(test_data_loader), total=dl_len):
            input_ids = batch['input_ids'].to(attacker.f.device)
            logits = attacker.auto_encoder_forward(input_ids)
            result = evaluate_attacker_rouge(tok, logits, batch)
            rouge_1 += result['rouge-1']['f']
            rouge_2 += result['rouge-2']['f']
            rouge_l_f1 += result['rouge-l']['f']
            rouge_l_p += result['rouge-l']['p']
            rouge_l_r += result['rouge-l']['r']

    print(
        f'Epoch {epc} Test Rouge_l_f1: {rouge_l_f1 / dl_len}')  # , Test2 Rouge_l_f1: {rouge_l_f1_x / dl_len if attacker2 else 0}')
    p = get_ae_save_path(llm, fsha_path, args)
    if rouge_l_f1 / dl_len > 0.1 and (epc + 1) % 5 == 0 and args.save_checkpoint:
        attacker.save_pretrained(p + f'epoch_{epc}_rouge_{rouge_l_f1 / dl_len:.4f}')
    # if args.log_to_wandb:
    #     log_dict = {'epoch': epc, 'test_rouge_1': rouge_1 / dl_len, 'test_rouge_2': rouge_2 / dl_len,
    #                 'test_rouge_l_f1': rouge_l_f1 / dl_len, 'test_rouge_l_p': rouge_l_p / dl_len,
    #                 'test_rouge_l_r': rouge_l_r / dl_len}
    #     wandb.log(log_dict)
    # md.train(True)
    attacker.train(True)
    return rouge_1 / dl_len, rouge_2 / dl_len, rouge_l_f1 / dl_len, rouge_l_p / dl_len, rouge_l_r / dl_len


@dataclasses.dataclass
class AutoEncoderConfig(PretrainedConfig):
    vocab_size: int = 0
    n_embed: int = 0
    d_hidden_size: int = 256
    f_hidden_size: int = 256
    f_dropout: float = 0.1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FSHAAttacker(PreTrainedModel):
    config_class = AutoEncoderConfig

    def __init__(self, config: AutoEncoderConfig = None, target_config=None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.f_inv = GRUDRInverter(LSTMDRAttackerConfig(hidden_size=config.f_hidden_size, dropout=config.f_dropout,
                                                        vocab_size=config.vocab_size, n_embed=config.n_embed, ),
                                   target_config)
        self.config.vocab_size = self.f_inv.config.vocab_size
        self.config.n_embed = self.f_inv.config.n_embed
        self.f = GRUDRInverter(LSTMDRAttackerConfig(vocab_size=self.config.n_embed, n_embed=target_config.vocab_size,
                                                    hidden_size=self.config.f_hidden_size,
                                                    dropout=self.config.f_dropout))
        self.d = torch.nn.GRU(self.config.n_embed, self.config.d_hidden_size, batch_first=True)
        self.d_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.config.d_hidden_size, 1),
            # torch.nn.Sigmoid()
        )

    def f_forward(self, input_ids):
        input_ids.to(self.f.device)
        input_ids = torch.nn.functional.one_hot(input_ids, num_classes=self.config.vocab_size).float()
        return self.f(input_ids)

    def f_inv_forward(self, hidden_states):
        return self.f_inv(hidden_states)

    def auto_encoder_forward(self, input_ids):
        return self.f_inv_forward(self.f_forward(input_ids))

    def d_forward(self, hidden_states):
        # hidden_states (batch_size, seq_len, n_embed)
        hidden, _ = self.d(hidden_states)
        # output (batch_sze, 1)
        hidden = torch.dropout(hidden, p=0.1, train=self.training)
        # activation
        hidden = torch.relu(hidden)
        hidden = torch.mean(hidden, dim=1)
        return self.d_mlp(hidden)

    def fit_auto_encoder(self, llm, tokenizer, data_loader, test_loader, epochs=50, args=None):
        # assert llm.fl_config is not None and llm.fl_config.attack_mode
        optimizer = torch.optim.Adam(list(self.f.parameters()) + list(self.f_inv.parameters()), lr=1e-3,
                                     weight_decay=1e-5)
        with tqdm(total=epochs * len(data_loader)) as pbar:
            for epc in range(epochs):
                for batch in data_loader:
                    optimizer.zero_grad()
                    input_ids = batch['input_ids'].to(self.f.device)
                    # convert index to one-hot
                    out = self.auto_encoder_forward(input_ids)
                    # calculate mse of out and input_ids
                    loss = calc_unshift_loss(out, input_ids)
                    # loss = torch.nn.functional.mse_loss(out, torch.nn.functional.one_hot(input_ids, num_classes=self.vocab_size).float())
                    loss.backward()
                    optimizer.step()
                    pbar.set_description(f'AutoEncoder-Epoch {epc}')
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update()
                if (epc + 1) % 5 == 0:
                    evaluate_ae(epc, llm, self, tokenizer, test_loader, args)

            # input_att_mask = batch['input_att_mask'].to(self.device)
            # inter = llm(input_ids, attention_mask=input_att_mask)
