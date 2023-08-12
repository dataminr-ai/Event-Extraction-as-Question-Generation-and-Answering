from __future__ import absolute_import, division, print_function
import time
import os
import sys
import argparse
import logging

import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    BartTokenizer,
    BartForConditionalGeneration,
    BartConfig,
)

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

from utils import loader
from utils.evaluate import evaluate_qa
from utils.processors import dataset_processor

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S %Z",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end Event Extraction Evaluation")
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
        default="./model_checkpoint/trigger_model/"
                "trigger_predictions.json",
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
        default="./model_checkpoint/qg_model_t5",
        help="path to the trained QG model checkpoint",
    )
    parser.add_argument(
        "--eae_model_path",
        default="./model_checkpoint/eae_model_t5",
        help="path to the trained EAE model checkpoint",
    )
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument(
        "--gpu", type=int, default=0, help="gup id, set to -1 if use cpu mode"
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
    parser.add_argument(
        "--transformer_name",
        default="t5-large",
        help="name of the transformer backbone",
    )
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
    elif args.transformer_name == "bart-large":
        model_name = "facebook/bart-large"
        model_class = BartForConditionalGeneration
        tokenizer_mame = BartTokenizer
        config_name = BartConfig

    eae_config = config_name.from_pretrained(model_name)
    eae_model = model_class.from_pretrained(args.eae_model_path)
    eae_tokenizer = tokenizer_mame.from_pretrained(args.eae_model_path)
    pad_token_id = eae_tokenizer.pad_token_id
    args.hidden_dim = eae_config.d_model

    eae_model.eval()

    if eae_model.config.decoder_start_token_id is None:
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
    qg_config = qg_config_name.from_pretrained(model_name)
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
        tokenizer=eae_tokenizer,
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

    if args.gpu >= 0:
        eae_model.cuda()

    total_loss = 0
    total_steps = 0
    global_step = 0
    start_time = time.time()
    best_result = None

    test_result, test_preds = evaluate_qa(
        args,
        eae_tokenizer,
        eae_model,
        test_dataset_loader,
        processor.test_samples,
        processor.gold_samples,
        processor.test_features,
        multi_arg=args.multi_arg,
        qa_length_penalty=args.qa_length_penalty,
        lower_case=args.lower_case,
    )
    logger.info(
        f"test: p_c: {test_result['prec_c']:.2f}, "
        f"r_c: {test_result['recall_c']:.2f}, "
        f"f1_c: {test_result['f1_c']:.2f}, "
        f"p_i: {test_result['prec_i']:.2f}, "
        f"r_i: {test_result['recall_i']:.2f}, "
        f"f1_i: {test_result['f1_i']:.2f}"
    )
