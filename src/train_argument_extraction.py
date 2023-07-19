from __future__ import absolute_import, division, print_function
import time
import os
import sys
import math
import argparse
import json
import logging

import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    BartTokenizer,
    BartForConditionalGeneration,
    BartConfig,
    AdamW,
    Adafactor,
    get_linear_schedule_with_warmup,
)

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

from utils import loader
from utils.evaluate import evaluate_qa
from utils.processors import dataset_processor
from model.label_smoother_sum import LabelSmoother

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S %Z",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event Argument Extraction")
    parser.add_argument(
        "--hidden_dim", type=int, default=160, help="hidden dimention of lstm"
    )
    parser.add_argument("--lower_case", action="store_true", help="lower case")
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
        "--gold_file",
        default="./data_process/data/ace-event/processed-data/"
                "default-settings/json/test_convert.json",
        help="path to test bio file",
    )
    parser.add_argument(
        "--event_template_doc",
        default="./ACE_templates/ACE_dynamic_templates.tsv",
        help="path to ACE event template file",
    )
    parser.add_argument(
        "--qg_model_path",
        default="./model_checkpoint/qg_model",
        help="path to the trained QG model checkpoint",
    )
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument(
        "--models_dir", default="./model_checkpoint/eae_model",
        help="directory of checkpoint"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="gup id, set to -1 if use cpu mode"
    )
    parser.add_argument(
        "--opt",
        choices=["adam", "adafactor"],
        default="adafactor",
        help="optimizer [adafactor,adam]",
    )
    parser.add_argument("--lr", type=float, default=0.015,
                        help="learning rate")
    parser.add_argument(
        "--lr_decay", type=float, default=1e-5,
        help="decay ratio of learning rate"
    )
    parser.add_argument("--epoch", type=int, default=20,
                        help="number of epoches")
    parser.add_argument("--clip_grad", type=float, default=1.0,
                        help="grad clip at")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
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
    parser.add_argument(
        "--eval_step",
        default=100,
        type=int,
        help="number of steps between two evaluation",
    )
    parser.add_argument("--eval_metric", default="f1_c", type=str)
    parser.add_argument("--normalize", action="store_true",
                        help="normalize digits")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--multi_arg",
        action="store_true",
        help="predict multiple args for each arg role",
    )
    parser.add_argument(
        "--na_format",
        choices=["empty", "na_format"],
        default="empty",
        help="empty or na_format"
    )
    parser.add_argument("--label_smoothing_factor", default=0.0, type=float)
    parser.add_argument(
        "--transformer_name",
        default="t5-large",
        help="name of the transformer backbone",
    )
    parser.add_argument("--run_id", default="", help="name of the dataset")
    parser.add_argument("--qg_model_type", default="t5",
                        help="name of the dataset")
    parser.add_argument(
        "--qg_length_penalty", default=0.0, type=float,
        help="name of the dataset"
    )
    parser.add_argument(
        "--qa_length_penalty", default=-2.5, type=float,
        help="name of the dataset"
    )
    parser.add_argument(
        "--correct_bias",
        action="store_true",
        help="predict multiple args for each arg role",
    )
    args = parser.parse_args()

    if args.seed is not None:
        loader.set_seed(args.seed)

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    processor = dataset_processor(args)

    arg_question_dic = loader.load_template_t5(args.event_template_doc)

    logger.info("loading dataset")
    processor.load_data(
        args.train_file, args.dev_file, args.test_file,
        gold_path=args.gold_file
    )

    # define model
    if args.transformer_name == "t5-large":
        model_name = "t5-large"
        model_class = T5ForConditionalGeneration
        tokenizer_mame = T5Tokenizer
        config_name = T5Config
        config = config_name.from_pretrained(model_name)
        tokenizer = tokenizer_mame.from_pretrained(model_name)
        model = model_class.from_pretrained(
            model_name, cache_dir="./pre-trained-model-cache"
        )
        pad_token_id = tokenizer.pad_token_id
        args.dropout = config.dropout_rate
        args.hidden_dim = config.d_model
    elif args.transformer_name == "bart-large":
        model_name = "facebook/bart-large"
        model_class = BartForConditionalGeneration
        tokenizer_mame = BartTokenizer
        config_name = BartConfig
        config = config_name.from_pretrained(model_name)
        tokenizer = tokenizer_mame.from_pretrained(model_name)
        model = model_class.from_pretrained(
            model_name, cache_dir="./pre-trained-model-cache"
        )
        pad_token_id = tokenizer.pad_token_id
        args.dropout = config.dropout
        args.hidden_dim = config.d_model

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    # load trained QG model checkpoint
    logger.info(f"loading {args.transformer_name} QG model checkpoint")
    if args.transformer_name == "t5-large":
        qg_model_class = T5ForConditionalGeneration
        qg_tokenizer_mame = T5Tokenizer
        qg_config_name = T5Config
    elif args.transformer_name == "bart-large":
        qg_model_class = BartForConditionalGeneration
        qg_tokenizer_mame = BartTokenizer
        qg_config_name = BartConfig
    qg_config = config_name.from_pretrained(model_name)
    qg_model = qg_model_class.from_pretrained(args.qg_model_path)
    qg_tokenizer = qg_tokenizer_mame.from_pretrained(args.qg_model_path)
    qg_model.eval()

    if args.seed is not None:
        loader.set_seed(args.seed)

    logger.info("generate features")
    (
        train_dataset_loader,
        dev_dataset_loader,
        test_dataset_loader,
    ) = processor.construct_dataset_qa(
        arg_question_dic,
        tokenizer=tokenizer,
        multi_arg=args.multi_arg,
        qg_model=qg_model,
        na_format=args.na_format,
        qg_tokenizer=qg_tokenizer,
        transformer_name=args.transformer_name,
        qg_length_penalty=args.qg_length_penalty,
        lower_case=args.lower_case,
    )

    tot_len = len(train_dataset_loader)
    logger.info(f"{tot_len} batches")

    logger.info("initialize optimizer")
    # define optimizer
    if args.opt == "adam":
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
                math.ceil(
                    tot_len / args.gradient_accumulation_steps) * args.epoch
        )
        warmup_steps = int(
            args.warmup_proportion * num_train_optimization_steps)
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.lr,
            correct_bias=args.correct_bias
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_train_optimization_steps,
        )
    elif args.opt == "adafactor":
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
                math.ceil(
                    tot_len / args.gradient_accumulation_steps) * args.epoch
        )

        warmup_steps = int(
            args.warmup_proportion * num_train_optimization_steps)

        optimizer = Adafactor(
            model.parameters(),
            lr=args.lr,
            clip_threshold=args.clip_grad,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_train_optimization_steps,
        )

    label_smoother = LabelSmoother(epsilon=args.label_smoothing_factor)

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
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Start epoch #{}/{} (lr = {})...".format(epoch, args.epoch,
                                                     current_lr)
        )

        for step, batch in enumerate(train_dataset_loader):
            (
                batch_input_ids,
                batch_attention_masks,
                batch_output_ids,
                batch_feature_idx,
            ) = batch

            if args.gpu >= 0:
                batch_input_ids = batch_input_ids.cuda()
                batch_attention_masks = batch_attention_masks.cuda()
                batch_output_ids = batch_output_ids.cuda()

            model_output = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                labels=batch_output_ids,
            )
            loss = label_smoother(model_output, batch_output_ids)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            total_loss += loss.item()
            epoch_loss += loss.item()
            total_steps += 1

            loss.backward()

            if total_steps % args.gradient_accumulation_steps == 0:
                if args.opt != "adafactor":
                    nn.utils.clip_grad_norm_(model.parameters(),
                                             args.clip_grad)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.eval_step == 0:
                    save_model = False

                    dev_result, dev_preds = evaluate_qa(
                        args,
                        tokenizer,
                        model,
                        dev_dataset_loader,
                        processor.dev_samples,
                        processor.dev_samples,
                        processor.dev_features,
                        multi_arg=args.multi_arg,
                        qa_length_penalty=args.qa_length_penalty,
                        lower_case=args.lower_case,
                    )

                    if (best_result is None) or (
                            dev_result[args.eval_metric] > best_result[
                        args.eval_metric]
                    ):
                        best_result = dev_result
                        save_model = True

                        test_result, test_preds = evaluate_qa(
                            args,
                            tokenizer,
                            model,
                            test_dataset_loader,
                            processor.test_samples,
                            processor.gold_samples,
                            processor.test_features,
                            multi_arg=args.multi_arg,
                            qa_length_penalty=args.qa_length_penalty,
                            lower_case=args.lower_case,
                        )

                        if not os.path.exists(args.models_dir):
                            os.makedirs(args.models_dir)
                        with open(
                                os.path.join(args.models_dir,
                                             "arg_predictions.json"),
                                "w",
                        ) as writer:
                            for key in test_preds:
                                writer.write(
                                    json.dumps(test_preds[key],
                                               default=int) + "\n"
                                )

                        logger.info(
                            f"Epoch: {epoch}/{args.epoch}, "
                            f"Step: {step + 1} / {len(train_dataset_loader)}, "
                            f"used_time = {time.time() - start_time:.2f}s, "
                            f"loss = {total_loss / total_steps:.6f}"
                        )

                        logger.info(
                            f"!!! Best dev {args.eval_metric} "
                            f"(lr={optimizer.param_groups[0]['lr']:.8f}, "
                            f"epoch={epoch}): "
                            f"p_c: {dev_result['prec_c']:.2f}, "
                            f"r_c: {dev_result['recall_c']:.2f}, "
                            f"f1_c: {dev_result['f1_c']:.2f}, "
                            f"p_i: {dev_result['prec_i']:.2f}, "
                            f"r_i: {dev_result['recall_i']:.2f}, "
                            f"f1_i: {dev_result['f1_i']:.2f}, "
                            f"test: p_c: {test_result['prec_c']:.2f}, "
                            f"r_c: {test_result['recall_c']:.2f}, "
                            f"f1_c: {test_result['f1_c']:.2f}, "
                            f"p_i: {test_result['prec_i']:.2f}, "
                            f"r_i: {test_result['recall_i']:.2f}, "
                            f"f1_i: {test_result['f1_i']:.2f}"
                        )

                    if save_model:
                        model_to_save = (
                            model.module if hasattr(model,
                                                    "module") else model
                        )

                        if not os.path.exists(args.models_dir):
                            os.makedirs(args.models_dir)

                        output_model_file = os.path.join(
                            args.models_dir, "pytorch_model.bin"
                        )
                        output_config_file = os.path.join(
                            args.models_dir, "config.json"
                        )
                        torch.save(model_to_save.state_dict(),
                                   output_model_file)
                        model_to_save.config.to_json_file(
                            output_config_file)
                        tokenizer.save_vocabulary(args.models_dir)
                    model.train()
