from __future__ import absolute_import, division, print_function
import time
import os
import sys
import math
import argparse
import logging

import torch
import torch.nn as nn
import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, \
    BartTokenizer, BartForConditionalGeneration, BartConfig, \
    AdamW, get_linear_schedule_with_warmup, Adafactor
from datasets import load_metric

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

from utils import loader
from utils.evaluate import evaluate_qg
from utils.processors import dataset_processor

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S %Z')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QG for EE')
    parser.add_argument('--lower_case', action='store_true', help='lower case')
    parser.add_argument(
        "--train_file",
        default="./data_process/data/ace-event/processed-data/"
                "default-settings/json/train_convert.json",
        help="path to training file",
    )
    parser.add_argument(
        "--dev_file",
        default="./data_process/data/ace-event/processed-data/"
                "default-settings/json/dev_convert.json",
        help="path to dev file",
    )
    parser.add_argument(
        "--test_file",
        default="./data_process/data/ace-event/processed-data/"
                "default-settings/json/test_convert.json",
        help="path to test file",
    )
    parser.add_argument('--gold_file',
                        default="./data_process/data/ace-event/processed-data/"
                                "default-settings/json/test_convert.json",
                        help='path to test bio file')
    parser.add_argument(
        "--event_template_doc",
        default="./ACE_templates/ACE_dynamic_templates.tsv",
        help="path to ACE event template file",
    )
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument(
        "--models_dir",
        default="./model_checkpoint/qg_model",
        help="directory to save model checkpoint",
    )
    parser.add_argument('--gpu', type=int, default=0,
                        help='gup id, set to -1 if use cpu mode')
    parser.add_argument('--opt', choices=['adam', 'adafactor'],
                        default='adam', help='optimizer [adam, adafactor]')
    parser.add_argument('--lr', type=float, default=1e-6,
                        help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=1e-5,
                        help='decay ratio of learning rate')
    parser.add_argument("--epoch", type=int, default=20,
                        help="number of epochs")
    parser.add_argument("--clip_grad", type=float, default=1.0,
                        help="grad clip at")
    parser.add_argument('--transformer_name', default='t5-large',
                        help='name of the transformer backbone')
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32,
        help="Number of updates steps to accumulate "
             "before performing a backward/update pass.",
    )
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument(
        "--eval_step",
        default=100,
        type=int,
        help="number of steps between two evaluation",
    )
    parser.add_argument("--eval_metric", default='rouge', type=str)
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for random initialization"
    )
    parser.add_argument('--qg_length_penalty', default=0.0, type=float,
                        help='length penalty for decoder output generation')
    parser.add_argument(
        "--correct_bias",
        action="store_true",
        help="set the correct_bias of the optimizer as True when add argument",
    )
    args = parser.parse_args()
    transformers.logging.set_verbosity_error()

    if args.seed is not None:
        loader.set_seed(args.seed)

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    processor = dataset_processor(args)

    arg_question_dic = loader.load_template_t5(args.event_template_doc)

    processor.load_data(args.train_file, args.dev_file, args.test_file)

    # define model
    if args.transformer_name == 't5-large':
        model_name = 't5-large'
        model_class = T5ForConditionalGeneration
        tokenizer_mame = T5Tokenizer
        config_name = T5Config
        config = config_name.from_pretrained(
            model_name)
        tokenizer = tokenizer_mame.from_pretrained(
            model_name)
        model = model_class.from_pretrained(
            model_name,
            cache_dir='./pre-trained-model-cache')
        pad_token_id = tokenizer.pad_token_id
        logger.info(f'tokenizer.pad_token_id is {pad_token_id}')
        logger.info(f"Vocab size is {len(tokenizer.get_vocab())}")
        args.dropout = config.dropout_rate
        args.hidden_dim = config.d_model
    elif args.transformer_name == 'bart-large':  # google/t5-v1_1-base
        model_name = 'facebook/bart-large'
        model_class = BartForConditionalGeneration
        tokenizer_mame = BartTokenizer
        config_name = BartConfig
        config = config_name.from_pretrained(
            model_name)
        tokenizer = tokenizer_mame.from_pretrained(
            model_name)
        model = model_class.from_pretrained(
            model_name,
            cache_dir='./pre-trained-model-cache')
        pad_token_id = tokenizer.pad_token_id
        logger.info(f'tokenizer.pad_token_id is {pad_token_id}')
        logger.info(f"Vocab size is {len(tokenizer.get_vocab())}")
        args.dropout = config.dropout
        args.hidden_dim = config.d_model

    train_dataset_loader, dev_dataset_loader, test_dataset_loader = \
        processor.construct_dataset_qg(arg_question_dic,
                                       tokenizer=tokenizer,
                                       lower_case=args.lower_case)

    tot_len = len(train_dataset_loader)
    logger.info(f"{tot_len} batches")

    # define optimizer
    if args.opt == 'adam':
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': args.lr_decay},
            {'params': [p for n, p in param_optimizer if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        total_sample = sum([len(x) for x in train_dataset_loader])
        print('training sample: ', total_sample)
        num_train_optimization_steps = math.ceil(
            tot_len / args.gradient_accumulation_steps) * args.epoch
        warmup_steps = int(
            args.warmup_proportion * num_train_optimization_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,
                          correct_bias=args.correct_bias)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
    elif args.opt == 'adafactor':
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': args.lr_decay},
            {'params': [p for n, p in param_optimizer if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        num_train_optimization_steps = math.ceil(
            tot_len / args.gradient_accumulation_steps) * args.epoch
        warmup_steps = int(
            args.warmup_proportion * num_train_optimization_steps)

        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            clip_threshold=1.0
        )
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

    rouge = load_metric('rouge')
    bleu = load_metric('bleu')

    if args.gpu >= 0:
        model.cuda()

    total_loss = 0
    total_steps = 0
    global_step = 0
    start_time = time.time()
    best_result = None

    optimizer.zero_grad()
    for epoch in range(args.epoch):
        epoch_loss = 0
        model.train()

        for step, batch in enumerate(train_dataset_loader):
            batch_input_ids, batch_attention_masks, \
            batch_output_ids, batch_feature_idx = batch

            if args.gpu >= 0:
                batch_input_ids = batch_input_ids.cuda()
                batch_attention_masks = batch_attention_masks.cuda()
                batch_output_ids = batch_output_ids.cuda()

            loss = model(input_ids=batch_input_ids,
                         attention_mask=batch_attention_masks,
                         labels=batch_output_ids).loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            total_loss += loss.item()
            epoch_loss += loss.item()
            total_steps += 1
            loss.backward()

            if total_steps % args.gradient_accumulation_steps == 0:
                if args.opt != 'adafactor':
                    nn.utils.clip_grad_norm_(model.parameters(),
                                             args.clip_grad)

                optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()
                global_step += 1

                if global_step % args.eval_step == 0:
                    save_model = False
                    dev_result = evaluate_qg(args, tokenizer,
                                             model,
                                             dev_dataset_loader,
                                             rouge, bleu,
                                             length_penalty=args.qg_length_penalty)

                    if (best_result is None) or (
                            dev_result[args.eval_metric] > best_result[
                        args.eval_metric]):
                        best_result = dev_result
                        save_model = True
                        test_result = evaluate_qg(args,
                                                  tokenizer,
                                                  model,
                                                  test_dataset_loader,
                                                  rouge,
                                                  bleu,
                                                  length_penalty=args.qg_length_penalty)

                        logger.info(
                            f"Epoch: {epoch}, "
                            f"Step: {step + 1} / {len(train_dataset_loader)}, "
                            f"used_time = {time.time() - start_time:.2f}s, "
                            f"loss = {total_loss / total_steps:.6f}, "
                            f"dev rouge score = {dev_result['rouge']:.3f}, "
                            f"dev bleu score = {dev_result['bleu']:.3f}, "
                            f"test rouge score = {test_result['rouge']:.3f}, "
                            f"test bleu score = {test_result['bleu']:.3f}")

                    if save_model:
                        model_to_save = model.module if hasattr(
                            model, 'module') else model

                        if not os.path.exists(args.models_dir):
                            os.makedirs(args.models_dir)
                        output_model_file = os.path.join(
                            args.models_dir, "pytorch_model.bin")
                        output_config_file = os.path.join(args.models_dir,
                                                          "config.json")
                        torch.save(model_to_save.state_dict(),
                                   output_model_file)
                        model_to_save.config.to_json_file(
                            output_config_file)
                        tokenizer.save_vocabulary(args.models_dir)

                    model.train()
