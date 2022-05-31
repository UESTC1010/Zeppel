import types
import argparse
import logging
from functools import partial
import json
import Levenshtein
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    BertGenerationConfig,
    BertGenerationEncoder,
    BertTokenizer,
    EncoderDecoderModel,
    EncoderDecoderConfig,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    XLNetTokenizer,
    TFGPT2LMHeadModel,
)

from utils import TSVDataset, collect_fn, get_logger, build_inputs_with_special_tokens
import sacrebleu
import os
import random
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import jieba

class XLNetTokenizer(XLNetTokenizer):
    translator = str.maketrans(" \n", "\u2582\u2583")

    def _tokenize(self, text, *args, **kwargs):
        text = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        text = " ".join(text)
        return super()._tokenize(text, *args, **kwargs)

    def _decode(self, *args, **kwargs):
        text = super()._decode(*args, **kwargs)
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')
        return text

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.ngpu > 1:
        torch.cuda.manual_seed_all(args.seed)


def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    torch.distributed.barrier()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()
    return rt


def get_optimizer_and_schedule(args, model: EncoderDecoderModel):
    all_params = model.parameters()
    sp_params = []
    # 根据自己的筛选规则  将所有网络参数进行分组
    for pname, p in model.named_parameters():
        if pname=='conv.weight' or pname=='conv.bias':
            sp_params += [p]
    # 取回分组参数的id
    params_id = list(map(id, sp_params))
    # 取回剩余分特殊处置参数的id
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))

    # 预训练参数和初始化参数使用不同的学习率
    if args.ngpu > 1:
        model = model.module
    optimizer = optim.AdamW([{'params':other_params, 'lr':args.finetune_lr},
                            {'params':sp_params, 'lr':args.lr}])

    schedule = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=args.num_training_steps,
        num_warmup_steps=args.num_warmup_steps,
    )
    return optimizer, schedule


def get_model(args):
    if args.model_path:
        model = EncoderDecoderModel.from_pretrained(args.model_path)
        src_tokenizer = BertTokenizer.from_pretrained(
            os.path.join(args.model_path, "src_tokenizer")
        )
        tgt_tokenizer = BertTokenizer.from_pretrained(
            os.path.join(args.model_path, "tgt_tokenizer")
        )
        tgt_tokenizer.build_inputs_with_special_tokens = types.MethodType(
            build_inputs_with_special_tokens, tgt_tokenizer
        )
        if args.local_rank == 0 or args.local_rank == -1:
            print("model and tokenizer load from save success")
    else:
        src_tokenizer = BertTokenizer.from_pretrained(args.src_pretrain_dataset_name)
        tgt_tokenizer = BertTokenizer.from_pretrained(args.tgt_pretrain_dataset_name)
        tgt_tokenizer.add_special_tokens(
            {"bos_token": "[BOS]", "eos_token": "[EOS]", "pad_token": "[PAD]"}
        )
        tgt_tokenizer.build_inputs_with_special_tokens = types.MethodType(
            build_inputs_with_special_tokens, tgt_tokenizer
        )
        encoder = BertGenerationEncoder.from_pretrained(args.src_pretrain_dataset_name)
        decoder = GPT2LMHeadModel.from_pretrained(
            args.tgt_pretrain_dataset_name, is_decoder=True
        )
        decoder.resize_token_embeddings(len(tgt_tokenizer))
        decoder.config.bos_token_id = tgt_tokenizer.bos_token_id
        decoder.config.eos_token_id = tgt_tokenizer.eos_token_id
        decoder.config.vocab_size = len(tgt_tokenizer)
        decoder.config.is_decoder = True
        decoder.add_cross_attention = False
        model_config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder.config, decoder.config
        )
        model = EncoderDecoderModel(
            encoder=encoder, decoder=decoder, config=model_config
        )
    if args.local_rank != -1:
        model = model.to(device)
    if args.ngpu > 1:
        print("{}/{} GPU start".format(args.local_rank, torch.cuda.device_count()))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    optimizer, scheduler = get_optimizer_and_schedule(args, model)

    return model, src_tokenizer, tgt_tokenizer, optimizer, scheduler


def save_model(
    args,
    model,
    optimizer,
    src_tokenizer: BertTokenizer,
    tgt_tokenizer: GPT2Tokenizer,
    nstep,
    nepoch,
    bleu,
    loss,
):
    # 记录整体训练评价结果
    train_metric_log_file = os.path.join(args.output_dir, "training_metric.tsv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.exists(train_metric_log_file):
        with open(train_metric_log_file, "a", encoding="utf-8") as fa:
            fa.write("{}\t{}\t{}\t{}\n".format(nepoch, nstep, loss, bleu))
    else:
        with open(train_metric_log_file, "w", encoding="utf-8") as fw:
            fw.write("epoch\tstep\tloss\tbleu\n")
            fw.write("{}\t{}\t{}\t{}\n".format(nepoch, nstep, loss, bleu))

    # 保存模型
    model_save_path = os.path.join(
        args.output_dir, "epoch{}_step{}/".format(nepoch, nstep)
    )
    os.makedirs(model_save_path)
    model.save_pretrained(model_save_path)
    if args.local_rank == 0 or args.local_rank == -1:
        print(
            "epoch:{} step:{} loss:{} bleu:{} model save complete.".format(
                nepoch, nstep, round(loss, 4), round(bleu, 4)
            )
        )
    if args.save_optimizer:
        torch.save(optimizer, os.path.join(model_save_path, "optimizer.pt"))

    # 保存tokenizer
    src_tokenizer.save_pretrained(os.path.join(model_save_path, "src_tokenizer"))
    tgt_tokenizer.save_pretrained(os.path.join(model_save_path, "tgt_tokenizer"))


def main(args):
    if args.local_rank == 0 or args.local_rank == -1:
        print(vars(args))
    model, src_tokenizer, tgt_tokenizer, optimizer, scheduler = get_model(args)
    if args.ispredict:
        if args.eval_data_path:
            eval_dataset = TSVDataset(data_path=args.eval_data_path)
            print(
                "load eval_dataset:{}".format(
                   len(eval_dataset)
                )
            )
            collect_fn_ = partial(
                collect_fn,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                max_src_len=args.max_src_len,
                max_tgt_len=args.max_tgt_len,
            )
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=args.eval_batch_size,
                collate_fn=collect_fn_,
                num_workers=args.num_workers,
            )
            ref,hyp = eval(model,
                        eval_dataloader,
                        src_tokenizer,
                        tgt_tokenizer,
                        max_src_len=args.max_src_len,
                        eval_data_path=args.eval_data_path)
            out_path = args.output_dir + args.out_path
            with open(out_path,'a',encoding='utf-8') as w_fp:
                for ref_,hyp_ in zip(ref,hyp):
                    ref_, hyp_ = ref_.replace(' ',''),hyp_.replace(' ','')
                    w_fp.write(ref_+'\t'+hyp_+'\n')
            print('predict done!')
        else:
            while True:
                input_str = input("input src: ")
                output_str = predict(
                    input_str,
                    model,
                    src_tokenizer,
                    tgt_tokenizer,
                    args.max_src_len,
                    args.max_tgt_len,
                )
                # print(output_str)
                for i,out in enumerate(output_str):
                    distance = Levenshtein.distance(input_str,out) / len(input_str)
                    print(i,'\t',out,distance)
                print()
    else:
        if args.eval_data_path:
            train_dataset = TSVDataset(data_path=args.train_data_path)
            eval_dataset = TSVDataset(data_path=args.eval_data_path)
            if args.local_rank == 0 or args.local_rank == -1:
                print(
                    "load train_dataset:{} and eval_dataset:{}".format(
                        len(train_dataset), len(eval_dataset)
                    )
                )
        else:
            dataset = TSVDataset(data_path=args.train_data_path)
            # print("dataset:", dataset[0])
            train_size = int(args.train_dataset_ratio * len(dataset))
            eval_size = len(dataset) - train_size
            train_dataset, eval_dataset = torch.utils.data.random_split(
                dataset, [train_size, eval_size], generator=torch.Generator()
            )
            if args.local_rank == 0 or args.local_rank == -1:
                print(
                    "load dataset:{} split into train_dataset{} and eval_dataset:{}".format(
                        len(dataset), train_size, eval_size
                    )
                )
        # print("train_dataset:",len(train_dataset))
        # print("train_dataset:",train_dataset[0])
        collect_fn_ = partial(
            collect_fn,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_src_len=args.max_src_len,
            max_tgt_len=args.max_tgt_len,
        )
        if args.ngpu > 1:
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.train_batch_size,
                collate_fn=collect_fn_,
                num_workers=args.num_workers,
                drop_last=args.drop_last,
                sampler=DistributedSampler(train_dataset),
            )
        else:
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.train_batch_size,
                collate_fn=collect_fn_,
                num_workers=args.num_workers,
                shuffle=True
            )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            collate_fn=collect_fn_,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
        )
        train(
            args=args,
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
        )


def train(
    args,
    model: EncoderDecoderModel,
    train_dataloader,
    eval_dataloader,
    optimizer,
    scheduler,
    src_tokenizer,
    tgt_tokenizer,
):
    eval_bleu = -1
    for epoch in range(args.nepoch):
        step = 0
        total_batch = len(train_dataloader)
        tbar = tqdm(total=total_batch,desc='training...')
        for data in train_dataloader:
            # optimizer.zero_grad()
            (
                input_ids,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                labels,
            ) = data
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                decoder_input_ids=decoder_input_ids.to(device),
                decoder_attention_mask=decoder_attention_mask.to(device),
                labels=labels.to(device),
                return_dict=True,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            model.zero_grad()
            if args.local_rank == 0 or args.local_rank == -1:
                # print('.............')
                # print('loss1',loss.cpu().item())
                if args.local_rank != -1:
                    # reduced_loss = reduce_tensor(loss).cpu().item()
                    reduced_loss = loss.cpu().item()
                else:
                    reduced_loss = loss.cpu().item()
                finetune_lr = scheduler.get_lr()[0]
                tbar.set_postfix(bleu=round(eval_bleu, 4), loss=round(reduced_loss, 4), lr=round(finetune_lr, 6))
                tbar.update()
                # print(
                #     "\rstep:{}/{}, bleu:{} loss:{}, lr:{}".format(
                #         step,
                #         total_batch,
                #         round(eval_bleu, 4),
                #         round(reduced_loss, 4),
                #         round(finetune_lr, 6),
                #     ),
                #     end="",
                # )
                writer.add_scalar("loss", reduced_loss, int(step * (1 + epoch)))
                writer.add_scalar("finetune_lr", finetune_lr, int(step * (1 + epoch)))
                if step % args.save_step == 0 or step % total_batch == 0:
                    eval_bleu = eval(
                        model, eval_dataloader,src_tokenizer, tgt_tokenizer, args.max_src_len
                    )
                    writer.add_scalar("bleu", eval_bleu, int(step * (1 + epoch)))
                    model_to_save = model.module if hasattr(model, "module") else model
                    save_model(
                        args,
                        model_to_save,
                        optimizer,
                        src_tokenizer,
                        tgt_tokenizer,
                        step,
                        epoch,
                        eval_bleu,
                        reduced_loss,
                    )


def eval(
    model: EncoderDecoderModel,
    eval_dataloader,
    src_tokenizer,
    tgt_tokenizer,
    max_src_len,
    eval_data_path=None,
    num_beams=10,
    num_beam_groups=5,
    diversity_penalty=2.5,
):
    hyp, ref = [], []
    # is_trans = False
    with torch.no_grad():
        for data in tqdm(eval_dataloader):
            input_ids, attention_mask, decoder_input_ids, _, _ = data
            batch_size = len(input_ids)
            if hasattr(model, "module"):
                generate = model.module.generate
            else:
                generate = model.generate
            if not eval_data_path:
            # if not eval_data_path or is_trans:
                outputs = generate(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    max_length=max_src_len,
                    num_beams=num_beams,
                    bos_token_id=tgt_tokenizer.bos_token_id,
                    eos_token_id=tgt_tokenizer.eos_token_id,
                    use_cache=False,  # 必须为False
                )
                hypoth = tgt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                refer = tgt_tokenizer.batch_decode(
                    decoder_input_ids, skip_special_tokens=True
                )
            else:
                refer = src_tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True
                )
                inputs_for_dbs = tgt_tokenizer(
                    refer,
                    padding="max_length",
                    truncation=True,
                    max_length=max_src_len,
                    return_tensors="pt",
                )
                inputs_for_dbs = inputs_for_dbs.input_ids.to(device)
                outputs = generate(
                    input_ids=input_ids.to(device),
                    inputs_for_dbs=inputs_for_dbs,
                    attention_mask=attention_mask.to(device),
                    max_length=max_src_len,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    num_beam_groups=num_beam_groups,
                    diversity_penalty=diversity_penalty,
                    encoder_no_repeat_ngram_size=0,
                    bos_token_id=tgt_tokenizer.bos_token_id,
                    eos_token_id=tgt_tokenizer.eos_token_id,
                    use_cache=False,
                )
                output_str = tgt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                out_list = [output_str[i].replace(' ','') for i in range(len(output_str))]
                hypoth = []
                for b_id in range(batch_size):
                    batch_out = out_list[b_id*num_beams:(b_id+1)*num_beams]
                    input_str = refer[b_id].replace(' ', '')
                    a_flag = False
                    for out in batch_out:
                        if not a_flag:
                            distance = Levenshtein.distance(input_str, out) / len(input_str)
                            if distance > 0.25:
                                # print(distance)
                                hypoth.append(out)
                                a_flag = True
                    if not a_flag:
                        hypoth.append(out)
                    # print(input_str)
                    # print(batch_out)
                    # print(hypoth[-1])
                    # input()

            # print("ref:{}".format(refer[0]))
            # print("hyp:{}".format(hypoth[0]))

            hyp.extend(hypoth)
            ref.extend(refer)
    if not eval_data_path:
        bleu = sacrebleu.corpus_bleu(hyp, [ref])
        return bleu.score
    else:
        return ref,hyp


def predict(
    input_str,
    model: EncoderDecoderModel,
    src_tokenizer,
    tgt_tokenizer,
    max_src_len,
    max_tgt_len,
    num_beam=10,
    num_beam_groups=5,
    diversity_penalty=0.0,
):
    inputs = src_tokenizer(
        [input_str],
        padding="max_length",
        truncation=True,
        max_length=max_src_len,
        return_tensors="pt",
    )
    inputs_for_dbs = tgt_tokenizer(
        [input_str],
        padding="max_length",
        truncation=True,
        max_length=max_src_len,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    inputs_for_dbs = inputs_for_dbs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    if hasattr(model, "module"):
        generate = model.module.generate
    else:
        generate = model.generate
    outputs = generate(
        input_ids=input_ids,
        inputs_for_dbs=inputs_for_dbs,
        attention_mask=attention_mask,
        max_length=max_tgt_len,
        num_beams=num_beam,
        num_return_sequences=num_beam,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        encoder_no_repeat_ngram_size=0,
        bos_token_id=tgt_tokenizer.bos_token_id,
        eos_token_id=tgt_tokenizer.eos_token_id,
        use_cache=False,
    )
    # outputs = generate(
    #     input_ids=input_ids,
    #     attention_mask=attention_mask,
    #     max_length=max_tgt_len,
    #     num_beams=num_beam,
    #     num_return_sequences=num_beam,
    #     bos_token_id=tgt_tokenizer.bos_token_id,
    #     eos_token_id=tgt_tokenizer.eos_token_id,
    #     use_cache=False,
    # )
    output_str = tgt_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return [output_str[i].replace(' ','') for i in range(len(output_str))]
    # return [output_str[i*(num_beam//num_beam_groups)].replace(' ','') for i in range(num_beam_groups)]


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-dataset_name", default="origin", type=str)
    parse.add_argument("-src_pretrain_dataset_name", default=None, type=str)
    parse.add_argument("-tgt_pretrain_dataset_name", default=None, type=str)
    parse.add_argument("-train_data_path", default=None, type=str)
    parse.add_argument("-eval_data_path", default=None, type=str)
    parse.add_argument("-log_path", default=None, type=str)
    parse.add_argument("-run_path", default=None, type=str)
    parse.add_argument("-output_dir", default="../checkpoints/", type=str)
    parse.add_argument("-out_path", default="out.txt", type=str)
    parse.add_argument("-optimizer", default="adam", type=str)
    parse.add_argument("-lr", default=1e-7, type=float)
    parse.add_argument("-finetune_lr", default=1e-5, type=float)
    parse.add_argument("-ngpu", default=1, type=int)
    parse.add_argument("-seed", default=17, type=int)
    parse.add_argument("-max_src_len", default=128, type=int)
    parse.add_argument("-max_tgt_len", default=128, type=int)
    parse.add_argument("-check_step", default=10, type=int)
    parse.add_argument("-save_step", default=100, type=int)
    parse.add_argument("-num_training_steps", default=100, type=int)
    parse.add_argument("-num_warmup_steps", default=100, type=int)
    parse.add_argument("-nepoch", default=1, type=int)
    parse.add_argument("-num_workers", default=1, type=int)
    parse.add_argument("-train_batch_size", default=32, type=int)
    parse.add_argument("-eval_batch_size", default=32, type=int)
    parse.add_argument("-drop_last", default=False, action="store_true")
    parse.add_argument("-ispredict", action="store_true", default=False)
    parse.add_argument("-save_optimizer", action="store_true", default=False)
    parse.add_argument("-train_dataset_ratio", default=0.9999, type=float)
    parse.add_argument("-model_path", default=None, type=str)
    parse.add_argument("--local_rank", default=0, type=int)
    args = parse.parse_args()
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        args.local_rank = args.local_rank
        device = torch.device("cpu")
    if args.local_rank == 0 or args.local_rank == -1:
        sw_log_path = os.path.join(args.run_path, args.dataset_name)
        if not os.path.exists(sw_log_path):
            os.makedirs(sw_log_path)
        writer = SummaryWriter(sw_log_path)
    set_seed(args)
    main(args)
