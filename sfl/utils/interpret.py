import torch
from matplotlib import pyplot

from sfl.utils.model import FLConfigHolder, get_embedding_layer


def saliency_analysis_generative(llm, input_ids, max_length=32):
    llm.train(False)
    with FLConfigHolder(llm) as ch:
        llm.fl_config.collect_intermediates = False
        llm.fl_config.noise_mode = 'none'
        ch.change_config()
        saliency_stacks = []
        saliency_avgs = []
        generated = []
        for batch_iid in input_ids:
            iids = batch_iid.unsqueeze(0).to(llm.device)  # (1, seq_len)
            all_saliency = []
            cnt = 0
            while True:
                input_embeds = get_embedding_layer(llm)(iids)
                input_embeds.requires_grad = True
                outputs = llm(inputs_embeds=input_embeds)
                logits = outputs.logits  # (1,seq_len,vocab_size)
                next_token_id = logits.argmax(dim=-1)[:, -1]  # (1,)
                loss = torch.max(logits[:, -1, :], dim=-1).values.mean()
                loss.backward()
                saliency = torch.abs(input_embeds.grad).sum(dim=-1)  # (1, seq_len)
                saliency = saliency / saliency.max(dim=-1).values.unsqueeze(-1)
                all_saliency.append(saliency)
                iids = torch.cat([iids, next_token_id.unsqueeze(0)], dim=-1)
                cnt += 1
                if next_token_id.item() == llm.config.eos_token_id or cnt > max_length:
                    break
            all_saliency = [s[0, :batch_iid.size(-1)] for s in all_saliency]  # (1, seq_len)
            saliency_stack = torch.stack(all_saliency)  # (all, seq_len)
            saliency_stacks.append(saliency_stack.detach().cpu().numpy())
            saliency_avgs.append(saliency_stack.mean(dim=0))
            generated.append(iids[0, input_ids.size(1):])
        batch_saliency_avg = torch.stack(saliency_avgs)  # (batch_size, seq_len)
        llm.train(True)
    saliency = batch_saliency_avg.detach().cpu().numpy()
    return saliency, generated, saliency_stacks


def saliency_analysis_direct(llm, input_ids):
    llm.train(False)
    with FLConfigHolder(llm) as ch:
        llm.fl_config.collect_intermediates = False
        llm.fl_config.noise_mode = 'none'
        ch.change_config()
        saliency_stacks = []
        saliency_avgs = []
        for batch_iid in input_ids:
            all_saliency = []
            cnt = 0
            for token_idx in range(batch_iid.size(-1)):
                input_embeds = get_embedding_layer(llm)(batch_iid.unsqueeze(0).to(llm.device))
                input_embeds.requires_grad = True
                outputs = llm(inputs_embeds=input_embeds)
                logits = outputs.logits  # (1,seq_len,vocab_size)
                loss = torch.max(logits[:, token_idx, :], dim=-1).values.mean()
                # loss = outputs.loss
                loss.backward()
                saliency = torch.abs(input_embeds.grad).sum(dim=-1)  # (1, seq_len)
                saliency = saliency / saliency.max(dim=-1).values.unsqueeze(-1)
                all_saliency.append(saliency)
                cnt += 1
            all_saliency = [s[0, :batch_iid.size(-1)] for s in all_saliency]  # (1, seq_len)
            saliency_stack = torch.stack(all_saliency)  # (all, seq_len)
            saliency_stacks.append(saliency_stack.detach().cpu().numpy())
            saliency_avgs.append(saliency_stack.mean(dim=0))
        batch_saliency_avg = torch.stack(saliency_avgs)  # (batch_size, seq_len)
        llm.train(True)
    saliency = batch_saliency_avg.detach().cpu().numpy()
    return saliency, saliency_stacks


def saliency_analysis_decoder(llm, input_ids):
    llm.train(True)
    with FLConfigHolder(llm) as ch:
        bk_inter = {k: v for k, v in llm.intermediate_fx.items()}
        llm.fl_config.collect_intermediates = True
        llm.fl_config.noise_mode = 'none'
        ch.change_config()
        saliency_stacks = []
        saliency_avgs = []
        for batch_iid in input_ids:
            all_saliency = []
            cnt = 0
            for token_idx in range(batch_iid.size(-1)):
                outputs = llm(batch_iid.unsqueeze(0).to(llm.device))
                logits = outputs.logits  # (1,seq_len,vocab_size)
                loss = torch.max(logits[:, token_idx, :], dim=-1).values.mean()
                # loss = outputs.loss
                loss.backward()
                b2tr_inter, _, _ = llm.get_all_inter()
                saliency = torch.abs(b2tr_inter.grad).sum(dim=-1)  # (1, seq_len)
                saliency = saliency / saliency.max(dim=-1).values.unsqueeze(-1)
                all_saliency.append(saliency)
                cnt += 1
            all_saliency = [s[0, :batch_iid.size(-1)] for s in all_saliency]  # (1, seq_len)
            saliency_stack = torch.stack(all_saliency)  # (all, seq_len)
            saliency_stacks.append(saliency_stack.detach().cpu().numpy())
            saliency_avgs.append(saliency_stack.mean(dim=0))
        llm.intermediate_fx = bk_inter
        batch_saliency_avg = torch.stack(saliency_avgs)  # (batch_size, seq_len)
    saliency = batch_saliency_avg.detach().cpu().numpy()
    return saliency, saliency_stacks


def saliency_analysis_atk(llm, attacker, input_ids):
    llm.train(False)
    with FLConfigHolder(llm) as ch:
        llm.fl_config.collect_intermediates = False
        attacker.train(True)
        llm.fl_config.noise_mode = 'none'
        llm.fl_config.attack_mode = 'b2tr'
        ch.change_config()
        saliency_stacks = []
        saliency_avgs = []
        attacked = []
        for batch_iid in input_ids:
            all_saliency = []
            cnt = 0
            for token_idx in range(batch_iid.size(-1)):
                input_embeds = get_embedding_layer(llm)(batch_iid.unsqueeze(0).to(llm.device))
                input_embeds.requires_grad = True
                inter = llm(inputs_embeds=input_embeds)
                logits = attacker(inter)
                loss = torch.max(logits[:, token_idx, :], dim=-1).values.mean()
                loss.backward()
                saliency = torch.abs(input_embeds.grad).sum(dim=-1)  # (1, seq_len)
                saliency = saliency / saliency.max(dim=-1).values.unsqueeze(-1)
                all_saliency.append(saliency)
                cnt += 1
            all_saliency = [s[0, :batch_iid.size(-1)] for s in all_saliency]  # (1, seq_len)
            saliency_stack = torch.stack(all_saliency)  # (all, seq_len)
            saliency_stacks.append(saliency_stack.detach().cpu().numpy())
            saliency_avgs.append(saliency_stack.mean(dim=0))
            attacked.append(logits[0, :, :].argmax(dim=-1))
        batch_saliency_avg = torch.stack(saliency_avgs)  # (batch_size, seq_len)
        llm.train(True)
        attacker.train(False)
    saliency = batch_saliency_avg.detach().cpu().numpy()
    return saliency, attacked, saliency_stacks


def saliency_analysis_atk_mid(llm, attacker, input_ids):
    llm.train(False)
    with FLConfigHolder(llm) as ch:
        llm.fl_config.collect_intermediates = False
        attacker.train(True)
        llm.fl_config.noise_mode = 'none'
        llm.fl_config.attack_mode = 'b2tr'
        ch.change_config()
        saliency_stacks = []
        saliency_avgs = []
        attacked = []
        for batch_iid in input_ids:
            all_saliency = []
            cnt = 0
            inter = llm(batch_iid.unsqueeze(0).to(llm.device))
            for token_idx in range(batch_iid.size(-1)):
                inter = inter.clone().detach().requires_grad_(True)
                logits = attacker(inter)
                loss = torch.max(logits[:, token_idx, :], dim=-1).values.mean()
                loss.backward()
                saliency = torch.abs(inter.grad).sum(dim=-1)  # (1, seq_len)
                saliency = saliency / saliency.max(dim=-1).values.unsqueeze(-1)
                all_saliency.append(saliency)
                cnt += 1
            all_saliency = [s[0, :batch_iid.size(-1)] for s in all_saliency]  # (1, seq_len)
            saliency_stack = torch.stack(all_saliency)  # (all, seq_len)
            saliency_stacks.append(saliency_stack.detach().cpu().numpy())
            saliency_avgs.append(saliency_stack.mean(dim=0))
            attacked.append(logits[0, :, :].argmax(dim=-1))
        batch_saliency_avg = torch.stack(saliency_avgs)  # (batch_size, seq_len)
        llm.train(True)
        attacker.train(False)
    saliency = batch_saliency_avg.detach().cpu().numpy()
    return saliency, attacked, saliency_stacks


def draw_saliency_map(saliency_matrix, input_sentence, output_sentence):
    # plot heatmap on saliency_matrix and log to wandb
    fig, ax = pyplot.subplots()
    fig.set_size_inches(16, 16)
    cax = ax.matshow(saliency_matrix, cmap='hot', vmin=0, vmax=1)
    # scale it to square
    ax.set_aspect('auto')
    fig.colorbar(cax)

    ax.set_xticks(ticks=range(len(input_sentence)), labels=input_sentence)
    ax.set_yticks(ticks=range(len(output_sentence)), labels=output_sentence)

    # ax.set_yticklabels(self.tokenizer.convert_ids_to_tokens(a), rotation=45)
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_title('Saliency Matrix')
    return fig


def draw_generative_saliency_maps(tokenizer, input_sentence, next_token_ids, stacks):
    figs = []
    for i, (q, a, saliency_matrix) in enumerate(zip(input_sentence, next_token_ids, stacks)):
        fig = draw_saliency_map(saliency_matrix, tokenizer.convert_ids_to_tokens(q),
                                tokenizer.convert_ids_to_tokens(a))
        figs.append(fig)
        # close figure to avoid memory leak
    return figs


def draw_direct_saliency_maps(tokenizer, input_sentence, stacks):
    figs = []
    for i, (q, saliency_matrix) in enumerate(zip(input_sentence, stacks)):
        fig = draw_saliency_map(saliency_matrix, tokenizer.convert_ids_to_tokens(q),
                                tokenizer.convert_ids_to_tokens(q))
        figs.append(fig)
    return figs
