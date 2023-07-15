from __future__ import absolute_import, division, print_function
from collections import Counter
import json
import logging

import torch

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S %Z",
)

logger = logging.getLogger(__name__)
from utils import loader
from utils.loader import (
    convert_examples_to_features_sequence,
    convert_examples_to_features_qg,
    convert_examples_to_features_qa,
    my_collate_trigger_train,
    my_collate_qg_train_t5,
    my_collate_qg_train_bart,
    my_collate_qa_t5,
    my_collate_qa_bart,
)


class dataset_processor:
    def __init__(self, args):
        self.args = args
        self.train_sentences = []
        self.dev_sentences = []
        self.test_sentences = []
        self.train_annotations = []
        self.dev_annotations = []
        self.test_annotations = []
        self.train_map_lists = []
        self.dev_map_lists = []
        self.test_map_lists = []

    def load_data(self, train_path, dev_path, test_path, gold_path=None):
        self.train_samples = loader.load_sentences(train_path)
        self.dev_samples = loader.load_sentences(dev_path)
        self.test_samples = loader.load_sentences(test_path)
        if gold_path:
            self.gold_samples = loader.load_sentences(gold_path)

    def generate_vocab(self, files_list):
        self.category_to_index = dict()
        self.index_to_category = dict()
        self.counter_event = Counter()

        self.category_to_index["O"] = 0
        self.index_to_category[0] = "O"
        for file in files_list:
            with open(file) as f:
                for line in f:
                    example = json.loads(line)
                    labels = example["trigger_label"]
                    for label in labels:
                        if label == "O":
                            continue
                        event_type = label
                        self.counter_event[event_type] += 1
                        if event_type not in self.category_to_index:
                            index = len(self.category_to_index)
                            self.category_to_index[event_type] = index
                            self.index_to_category[index] = event_type

    def construct_dataset_trigger(self, tokenizer=None, lower_case=False):
        self.train_dataset, self.train_features = convert_examples_to_features_sequence(
            examples=self.train_samples,
            tokenizer=tokenizer,
            category_to_index=self.category_to_index,
            lower_case=lower_case,
        )

        self.dev_dataset, self.dev_features = convert_examples_to_features_sequence(
            examples=self.dev_samples,
            tokenizer=tokenizer,
            category_to_index=self.category_to_index,
            lower_case=lower_case,
        )

        self.test_dataset, self.test_features = convert_examples_to_features_sequence(
            examples=self.test_samples,
            tokenizer=tokenizer,
            category_to_index=self.category_to_index,
            lower_case=lower_case,
        )

        self.tokenizer = tokenizer

        self.train_dataset_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            self.args.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=my_collate_trigger_train,
        )
        self.dev_dataset_loader = torch.utils.data.DataLoader(
            self.dev_dataset,
            self.args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=my_collate_trigger_train,
        )
        self.test_dataset_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            self.args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=my_collate_trigger_train,
        )
        return (
            self.train_dataset_loader,
            self.dev_dataset_loader,
            self.test_dataset_loader,
        )

    def construct_dataset_qa(
        self,
        arg_question_dic,
        tokenizer=None,
        multi_arg=False,
        qg_model=None,
        na_format="no_answer",
        qg_tokenizer=None,
        transformer_name="t5-large",
        qg_length_penalty=1.0,
        lower_case=False,
    ):
        (
            self.train_dataset,
            self.train_features,
        ) = convert_examples_to_features_qa(
            examples=self.train_samples,
            tokenizer=tokenizer,
            query_templates=arg_question_dic,
            nth_query=0,
            is_training=True,
            multi_arg=multi_arg,
            qg_model=qg_model,
            na_format=na_format,
            qg_tokenizer=qg_tokenizer,
            transformer_name=transformer_name,
            qg_length_penalty=qg_length_penalty,
            lower_case=lower_case,
        )
        logger.info("convert features dev 1")
        (
            self.dev_dataset,
            self.dev_features,
        ) = convert_examples_to_features_qa(
            examples=self.dev_samples,
            tokenizer=tokenizer,
            query_templates=arg_question_dic,
            nth_query=0,
            is_training=False,
            multi_arg=multi_arg,
            qg_model=qg_model,
            na_format=na_format,
            qg_tokenizer=qg_tokenizer,
            transformer_name=transformer_name,
            qg_length_penalty=qg_length_penalty,
            lower_case=lower_case,
        )
        logger.info("convert features test 1")
        (
            self.test_dataset,
            self.test_features,
        ) = convert_examples_to_features_qa(
            examples=self.test_samples,
            tokenizer=tokenizer,
            query_templates=arg_question_dic,
            nth_query=0,
            is_training=False,
            multi_arg=multi_arg,
            qg_model=qg_model,
            na_format=na_format,
            qg_tokenizer=qg_tokenizer,
            transformer_name=transformer_name,
            qg_length_penalty=qg_length_penalty,
            lower_case=lower_case,
        )

        self.tokenizer = tokenizer

        logger.info(
            f"{len(self.train_features)} training samples, {len(self.dev_features)} dev samples, {len(self.test_features)} test samples"
        )
        if "t5" in self.args.transformer_name:
            self.train_dataset_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                self.args.batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=my_collate_qa_t5,
            )
            self.dev_dataset_loader = torch.utils.data.DataLoader(
                self.dev_dataset,
                self.args.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=my_collate_qa_t5,
            )
            self.test_dataset_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                self.args.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=my_collate_qa_t5,
            )
        else:
            self.train_dataset_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                self.args.batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=my_collate_qa_bart,
            )
            self.dev_dataset_loader = torch.utils.data.DataLoader(
                self.dev_dataset,
                self.args.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=my_collate_qa_bart,
            )
            self.test_dataset_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                self.args.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=my_collate_qa_bart,
            )
        return (
            self.train_dataset_loader,
            self.dev_dataset_loader,
            self.test_dataset_loader,
        )

    def construct_dataset_qg(
        self,
        arg_question_dic,
        tokenizer=None,
        lower_case=False,
    ):
        self.train_dataset = convert_examples_to_features_qg(
            examples=self.train_samples,
            tokenizer=tokenizer,
            query_templates=arg_question_dic,
            lower_case=lower_case,
        )

        self.dev_dataset = convert_examples_to_features_qg(
            examples=self.dev_samples,
            tokenizer=tokenizer,
            query_templates=arg_question_dic,
            lower_case=lower_case,
        )

        self.test_dataset = convert_examples_to_features_qg(
            examples=self.test_samples,
            tokenizer=tokenizer,
            query_templates=arg_question_dic,
            lower_case=lower_case,
        )

        self.tokenizer = tokenizer

        if "t5" in self.args.transformer_name:
            self.train_dataset_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                self.args.batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=my_collate_qg_train_t5,
            )
            self.dev_dataset_loader = torch.utils.data.DataLoader(
                self.dev_dataset,
                self.args.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=my_collate_qg_train_t5,
            )
            self.test_dataset_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                self.args.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=my_collate_qg_train_t5,
            )
        else:
            self.train_dataset_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                self.args.batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=my_collate_qg_train_bart,
            )
            self.dev_dataset_loader = torch.utils.data.DataLoader(
                self.dev_dataset,
                self.args.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=my_collate_qg_train_bart,
            )
            self.test_dataset_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                self.args.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=my_collate_qg_train_bart,
            )
        return (
            self.train_dataset_loader,
            self.dev_dataset_loader,
            self.test_dataset_loader,
        )
