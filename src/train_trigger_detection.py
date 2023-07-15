from __future__ import absolute_import, division, print_function
import time
import os
import sys
import math
import json

import argparse
import pickle
import logging
import torch
import torch.nn as nn
from transformers import (
    AlbertConfig,
    AlbertModel,
    AlbertTokenizerFast,
    get_linear_schedule_with_warmup,
    AdamW,
)

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

from utils import loader
from model.encoder import trigger_tagger
from utils.evaluate import evaluate_event_trigger
from utils.processors import dataset_processor

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S %Z",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event Trigger Detection")
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
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for random initialization"
    )
    parser.add_argument(
        "--lower_case", action="store_true",
        help="lower case the input sentences"
    )
    parser.add_argument(
        "--transformer_name",
        default="albert-xxlarge-v2",
        help="name of the transformer backbone",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument(
        "--models_dir",
        default="./model_checkpoint/trigger_model",
        help="directory to save model checkpoint",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="gup id, set to -1 if use cpu mode"
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument(
        "--lr_decay", type=float, default=1e-5,
        help="decay ratio of learning rate"
    )
    parser.add_argument(
        "--eval_per_epoch",
        default=3,
        type=int,
        help="number of evaluates to perform per epoch",
    )
    parser.add_argument("--epoch", type=int, default=20,
                        help="number of epochs")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate "
             "before performing a backward/update pass.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform "
             "linear learning rate warmup for. "
             "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument("--clip_grad", type=float, default=1.0,
                        help="grad clip at")
    parser.add_argument(
        "--eval_metric",
        default="f1_c",
        type=str,
        help="metric for evaluation, default is f1_c " "(f1 of event classification)",
    )
    parser.add_argument(
        "--correct_bias",
        action="store_true",
        help="set the correct_bias of the optimizer as True when add argument",
    )
    args = parser.parse_args()

    # set random seed
    if args.seed is not None:
        loader.set_seed(args.seed)

    # set gpu
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )

    processor = dataset_processor(args)

    # load ACE data
    processor.load_data(args.train_file, args.dev_file, args.test_file)
    processor.generate_vocab([args.train_file, args.dev_file, args.test_file])
    logger.info(
        f"{len(processor.train_sentences)} sentences in training set; "
        f"{len(processor.dev_sentences)} sentences in dev set; "
        f"{len(processor.test_sentences)} sentences in test set."
    )

    # define model
    model_name = args.transformer_name
    model_class = AlbertModel
    tokenizer_mame = AlbertTokenizerFast
    config_name = AlbertConfig
    bert_config = config_name.from_pretrained(model_name)
    tokenizer = tokenizer_mame.from_pretrained(model_name)
    input_encoder = model_class.from_pretrained(model_name, config=bert_config)
    vocab = tokenizer.get_vocab()

    # check padding token id
    pad_token_id = tokenizer.pad_token_id
    logger.info(f"pad_token_id is {pad_token_id}")

    args.dropout = bert_config.hidden_dropout_prob
    args.hidden_dim = bert_config.hidden_size

    model = trigger_tagger(
        input_encoder,
        args.hidden_dim,
        len(processor.category_to_index),
        label_padding_idx=0,
        dropout=args.dropout,
    )

    # prepare dataloader
    (
        train_dataset_loader,
        dev_dataset_loader,
        test_dataset_loader,
    ) = processor.construct_dataset_trigger(
        tokenizer=tokenizer,
        lower_case=args.lower_case,
    )

    # set number of steps to finish before evaluation on dev set
    eval_step = max(
        1,
        len(train_dataset_loader)
        // (args.gradient_accumulation_steps * args.eval_per_epoch),
    )
    tot_len = len(train_dataset_loader)
    logger.info(f"{tot_len} batches in training set.")

    # define optimizer
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if "pooler" not in n[0]]
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if
                not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.lr_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if
                any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    total_sample = sum([len(x[0]) for x in train_dataset_loader])
    logger.info(f"training sample: {total_sample}")

    num_train_optimization_steps = (
            math.ceil(tot_len / args.gradient_accumulation_steps) * args.epoch
    )
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.lr,
        correct_bias=args.correct_bias
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_train_optimization_steps,
    )

    if args.gpu >= 0:
        model.cuda()

    total_loss = 0
    total_steps = 0
    global_step = 0
    start_time = time.time()
    best_result = None

    optimizer.zero_grad()
    model.train()

    for epoch in range(args.epoch):
        epoch_loss = 0
        for step, batch in enumerate(train_dataset_loader):
            (
                batch_input_ids,
                batch_trigger_labels,
                batch_attention_masks,
                batch_token_type_ids,
                _,
                _,
            ) = batch

            if args.gpu >= 0:
                batch_input_ids = batch_input_ids.cuda()
                batch_trigger_labels = batch_trigger_labels.cuda()
                batch_attention_masks = batch_attention_masks.cuda()
                batch_token_type_ids = batch_token_type_ids.cuda()

            trigger_inputs = {
                "sentence_ids": batch_input_ids,
                "attention_mask": batch_attention_masks,
                "token_type_ids": batch_token_type_ids,
            }

            loss, _ = model.loss(trigger_inputs, batch_trigger_labels)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            total_loss += loss.item()
            epoch_loss += loss.item()
            total_steps += 1

            loss.backward()

            if total_steps % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % eval_step == 0:
                    save_model = False
                    dev_result, dev_preds = evaluate_event_trigger(
                        args,
                        model,
                        dev_dataset_loader,
                        processor.dev_features,
                        processor.dev_samples,
                        processor.index_to_category,
                    )

                    if (best_result is None) or (
                            dev_result[args.eval_metric] > best_result[
                        args.eval_metric]
                    ):
                        best_result = dev_result
                        save_model = True

                        test_result, test_preds = evaluate_event_trigger(
                            args,
                            model,
                            test_dataset_loader,
                            processor.test_features,
                            processor.test_samples,
                            processor.index_to_category,
                        )

                        # print out predictions on test set
                        if not os.path.exists(args.models_dir):
                            os.makedirs(args.models_dir)
                        with open(
                                os.path.join(args.models_dir,
                                             "trigger_predictions.json"),
                                "w",
                        ) as writer:
                            for line in test_preds:
                                writer.write(
                                    json.dumps(line, default=int) + "\n")

                        logger.info(
                            f"Epoch: {epoch}, Step: "
                            f"{step + 1} / {len(train_dataset_loader)}, "
                            f"used_time = {time.time() - start_time:.2f}s, "
                            f"loss = {total_loss / total_steps:.6f}"
                        )
                        logger.info(
                            f"!!! Best dev {args.eval_metric} "
                            f"(lr={optimizer.param_groups[0]['lr']}, "
                            f"epoch={epoch}): dev: "
                            f"p_i: {dev_result['pre_i']:.2f}, "
                            f"r_i: {dev_result['rec_i']:.2f}, "
                            f"f1_i: {dev_result['f1_i']:.2f}, "
                            f"p_c: {dev_result['pre_c']:.2f}, "
                            f"r_c: {dev_result['rec_c']:.2f}, "
                            f"f1_c: {dev_result['f1_c']:.2f}, "
                            f"test: "
                            f"p_i: {test_result['pre_i']:.2f}, "
                            f"r_i: {test_result['rec_i']:.2f}, "
                            f"f1_i: {test_result['f1_i']:.2f}"
                            f"p_c: {test_result['pre_c']:.2f}, "
                            f"r_c: {test_result['rec_c']:.2f}, "
                            f"f1_c: {test_result['f1_c']:.2f}, "
                        )

                    if save_model:
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )

                        if not os.path.exists(args.models_dir):
                            os.makedirs(args.models_dir)
                        output_model_file = os.path.join(
                            args.models_dir, "pytorch_model.bin"
                        )
                        torch.save(model_to_save.state_dict(),
                                   output_model_file)

                        output_config_file = os.path.join(
                            args.models_dir, "config.json"
                        )

                        tokenizer.save_vocabulary(args.models_dir)

                        arg_save_list = {"args": vars(args)}

                        with open(
                                os.path.join(args.models_dir,
                                             "arg_save_list.pkl"), "wb"
                        ) as f:
                            pickle.dump(arg_save_list, f)

                    model.train()
