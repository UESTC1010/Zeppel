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
        if 'encoder' in pname:
            p.requires_grad=False
    # 取回分组参数的id
    params_id = list(map(id, sp_params))
    # 取回剩余分特殊处置参数的id
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    train_params = filter(lambda x: x.requires_grad is not False ,model.parameters())

    # 预训练参数和初始化参数使用不同的学习率
    if args.ngpu > 1:
        model = model.module
    optimizer = optim.AdamW([{'params':train_params, 'lr':args.finetune_lr},
                            {'params':sp_params, 'lr':args.lr}])
    # for pname, p in model.named_parameters():
    #     print(pname,p.requires_grad)
    # input()

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


def main(args):
    if args.local_rank == 0 or args.local_rank == -1:
        print(vars(args))
    model, src_tokenizer, tgt_tokenizer, optimizer, scheduler = get_model(args)
    if args.eval_data_path and not args.isTrans:
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
        temp = args.model_path.split('/')[-1]
        out_path = args.out_path + temp + '_paraEvalRes.txt'
        with open(out_path,'w',encoding='utf-8') as w_fp:
            for ref_,hyp_ in zip(ref,hyp):
                ref_, hyp_ = ref_.replace(' ',''),hyp_.replace(' ','')
                w_fp.write(ref_+'\t'+hyp_+'\n')
        print('predict done!')
    elif args.isTrans:
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
        ref, hyp = eval(model,
                        eval_dataloader,
                        src_tokenizer,
                        tgt_tokenizer,
                        max_src_len=args.max_src_len,
                        eval_data_path=args.eval_data_path,
                        isTrans=args.isTrans)
        temp = args.model_path.split('/')[-1]
        out_path = args.out_path + temp + '_transEvalRes.txt'
        with open(out_path, 'w', encoding='utf-8') as w_fp:
            for ref_, hyp_ in zip(ref, hyp):
                ref_, hyp_ = ref_.replace(' ', ''), hyp_.replace(' ', '')
                w_fp.write(ref_ + '\t' + hyp_ + '\n')
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
    isTrans=False
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
            if isTrans:
                inputs_refer = src_tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True
                )
                refer = tgt_tokenizer.batch_decode(
                    decoder_input_ids, skip_special_tokens=True
                )
                inputs_for_dbs = tgt_tokenizer(
                    inputs_refer,
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
                    encoder_no_repeat_ngram_size=5,
                    bos_token_id=tgt_tokenizer.bos_token_id,
                    eos_token_id=tgt_tokenizer.eos_token_id,
                    use_cache=False,
                )
                output_str = tgt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                out_list = [output_str[i].replace(' ', '') for i in range(len(output_str))]
                hypoth = []
                for b_id in range(batch_size):
                    batch_out = out_list[b_id * num_beams:(b_id + 1) * num_beams]
                    # 取每个beam的第一句
                    hypoth.append(batch_out[0])
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
                    encoder_no_repeat_ngram_size=5,
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
                    # 取每个beam的第一句
                    hypoth.append(batch_out[0])
                    # for out in batch_out:
                    #     if not a_flag:
                    #         distance = Levenshtein.distance(input_str, out) / len(input_str)
                    #         if distance > 0.25:
                    #             # print(distance)
                    #             hypoth.append(out)
                    #             a_flag = True
                    # if not a_flag:
                    #     hypoth.append(out)

            hyp.extend(hypoth)
            ref.extend(refer)
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
    diversity_penalty=0,
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
    parse.add_argument("-isTrans",  type=bool, default=False)
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
    args.isTrans = True if args.isTrans == 'True' else False
    set_seed(args)
    main(args)
