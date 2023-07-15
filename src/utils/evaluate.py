from __future__ import absolute_import, division, print_function
import sys
import copy
import collections

sys.path.append(".")
sys.path.append("..")

import torch
from utils import loader


def evaluate_event_trigger(
    args, model, eval_dataloader, eval_features, eval_examples, index_to_category
):
    model.eval()

    # get predictions
    gold_triggers = dict()
    pred_triggers = dict()
    for idx, batch in enumerate(eval_dataloader):
        (
            batch_input_ids,
            batch_trigger_labels,
            batch_attention_masks,
            batch_token_type_ids,
            batch_feature_idx,
            batch_trigger_masks,
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

        with torch.no_grad():
            preds, logits = model.decode(trigger_inputs)

        for i, (pred_list, label_list, feature_idx, trigger_mask) in enumerate(
            zip(preds, batch_trigger_labels, batch_feature_idx, batch_trigger_masks)
        ):
            pred_list = pred_list.detach().cpu().tolist()
            label_list = label_list.detach().cpu().tolist()
            trigger_mask = trigger_mask.detach().cpu().tolist()
            feature_idx = feature_idx.cpu().item()
            sentence_id = eval_features[feature_idx].example_id

            if sentence_id not in gold_triggers:
                gold_triggers[sentence_id] = []
            if sentence_id not in pred_triggers:
                pred_triggers[sentence_id] = []

            original_word_index = 0
            pre_gold_tag = None
            pre_sys_tag = None
            for i_token, (pre_token, label_token, one_trigger_mask) in enumerate(
                zip(pred_list, label_list, trigger_mask)
            ):
                # skip the non-leading sub-token
                if one_trigger_mask == 0:
                    continue

                # BIO tag post-processing, avoid the cases such as
                # 'B-injured I-demonstrate'
                if label_token != 0:
                    cur_gold_tag = index_to_category[label_token]
                    if pre_gold_tag is None:
                        cur_gold_tag = cur_gold_tag.replace("I-", "B-")
                        gold_triggers[sentence_id].append(
                            [original_word_index, original_word_index, cur_gold_tag[2:]]
                        )
                    elif cur_gold_tag[:2] == "B-":
                        gold_triggers[sentence_id].append(
                            [original_word_index, original_word_index, cur_gold_tag[2:]]
                        )
                    elif (
                        cur_gold_tag[:2] == "I-"
                        and cur_gold_tag[2:] == pre_gold_tag[2:]
                    ):
                        gold_triggers[sentence_id][-1][1] = original_word_index
                    else:
                        cur_gold_tag = cur_gold_tag.replace("I-", "B-")
                        gold_triggers[sentence_id].append(
                            [original_word_index, original_word_index, cur_gold_tag[2:]]
                        )
                    pre_gold_tag = cur_gold_tag

                if pre_token != 0:
                    cur_sys_tag = index_to_category[pre_token]
                    if pre_sys_tag is None:
                        cur_sys_tag = cur_sys_tag.replace("I-", "B-")
                        pred_triggers[sentence_id].append(
                            [original_word_index, original_word_index, cur_sys_tag[2:]]
                        )
                    elif cur_sys_tag[:2] == "B-":
                        pred_triggers[sentence_id].append(
                            [original_word_index, original_word_index, cur_sys_tag[2:]]
                        )
                    elif cur_sys_tag[:2] == "I-" and cur_sys_tag[2:] == pre_sys_tag[2:]:
                        pred_triggers[sentence_id][-1][1] = original_word_index
                    else:
                        cur_sys_tag = cur_sys_tag.replace("I-", "B-")
                        pred_triggers[sentence_id].append(
                            [original_word_index, original_word_index, cur_sys_tag[2:]]
                        )
                    pre_sys_tag = cur_sys_tag
                original_word_index += 1

    # get results (classification)
    gold_trigger_n, pred_trigger_n, true_positive_n = 0, 0, 0
    for sentence_id in pred_triggers:
        gold_sentence_triggers = gold_triggers[sentence_id]
        pred_sentence_triggers = pred_triggers[sentence_id]

        for _ in pred_sentence_triggers:
            pred_trigger_n += 1
        for _ in gold_sentence_triggers:
            gold_trigger_n += 1
        for trigger in pred_sentence_triggers:
            if trigger in gold_sentence_triggers:
                true_positive_n += 1

    if pred_trigger_n != 0:
        prec_c = 100.0 * true_positive_n / pred_trigger_n
    else:
        prec_c = 0
    if gold_trigger_n != 0:
        recall_c = 100.0 * true_positive_n / gold_trigger_n
    else:
        recall_c = 0
    if prec_c or recall_c:
        f1_c = 2 * prec_c * recall_c / (prec_c + recall_c)
    else:
        f1_c = 0

    # get results (identification)
    gold_triggers_offset = {}
    for sentence_id in gold_triggers:
        sent_triggers_offset = set()
        for trigger in gold_triggers[sentence_id]:
            sent_triggers_offset.add(trigger[0])
        gold_triggers_offset[sentence_id] = sent_triggers_offset

    pred_triggers_offset = {}
    for sentence_id in pred_triggers:
        sent_triggers_offset = set()
        for trigger in pred_triggers[sentence_id]:
            sent_triggers_offset.add(trigger[0])
        pred_triggers_offset[sentence_id] = sent_triggers_offset

    gold_trigger_n, pred_trigger_n, true_positive_n = 0, 0, 0
    for sentence_id in pred_triggers_offset:
        gold_sentence_triggers = gold_triggers_offset[sentence_id]
        pred_sentence_triggers = pred_triggers_offset[sentence_id]

        for _ in pred_sentence_triggers:
            pred_trigger_n += 1
        for _ in gold_sentence_triggers:
            gold_trigger_n += 1

        for trigger in pred_sentence_triggers:
            if trigger in gold_sentence_triggers:
                true_positive_n += 1

    if pred_trigger_n != 0:
        prec_i = 100.0 * true_positive_n / pred_trigger_n
    else:
        prec_i = 0
    if gold_trigger_n != 0:
        recall_i = 100.0 * true_positive_n / gold_trigger_n
    else:
        recall_i = 0
    if prec_i or recall_i:
        f1_i = 2 * prec_i * recall_i / (prec_i + recall_i)
    else:
        f1_i = 0
    result = collections.OrderedDict(
        [
            ("pre_c", prec_c),
            ("rec_c", recall_c),
            ("f1_c", f1_c),
            ("pre_i", prec_i),
            ("rec_i", recall_i),
            ("f1_i", f1_i),
        ]
    )

    trigger_preds = []

    for sentence_id, pred in enumerate(eval_examples):
        s_start = pred.s_start
        char_offset = pred.char_offset
        trigger_label = pred.trigger_label
        sentence = pred.sentence
        tokens = pred.tokens
        pre_test_sample = {
            "s_start": s_start,
            "sentence": sentence,
            "tokens": tokens,
            "trigger_label": trigger_label,
            "char_offset": char_offset,
            "event": [],
        }
        pred_sentence_triggers = pred_triggers[sentence_id]
        for trigger in pred_sentence_triggers:
            offset_start = char_offset[trigger[0]]
            offset_end = char_offset[trigger[1]] + len(tokens[trigger[1]]) - 1
            category = trigger[2]
            pre_test_sample["event"].append(
                [
                    [
                        offset_start,
                        offset_end,
                        category,
                        sentence[offset_start : offset_end + 1],
                    ]
                ]
            )
        trigger_preds.append(pre_test_sample)
    return result, trigger_preds


def evaluate_qg(
    args,
    tokenizer,
    model,
    eval_dataloader,
    rouge,
    bleu,
    length_penalty=1.0,
    lower_case=False,
):
    model.eval()
    all_predictions = []
    all_references = []
    for idx, batch in enumerate(eval_dataloader):
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

        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                num_beams=4,
                early_stopping=False,
                length_penalty=length_penalty,
            )

        for i in range(len(batch_feature_idx)):
            sys_answer = tokenizer.decode(outputs[i], skip_special_tokens=True)
            target_id_list = batch_output_ids[i]
            target_id_list[target_id_list == -100] = 0
            gold_answer = tokenizer.decode(target_id_list, skip_special_tokens=True)
            all_predictions.append(sys_answer)
            all_references.append(gold_answer)

    all_references_bleu = [[x.split()] for x in all_references]
    all_predictions_bleu = [x.split() for x in all_predictions]
    r_score = rouge.compute(predictions=all_predictions, references=all_references)

    b_score = bleu.compute(
        predictions=all_predictions_bleu, references=all_references_bleu
    )
    result = collections.OrderedDict(
        [("rouge", r_score["rouge1"].mid.fmeasure), ("bleu", b_score["bleu"])]
    )
    return result


def evaluate_qa(
    args,
    tokenizer,
    model,
    eval_dataloader,
    eval_examples,
    gold_examples,
    eval_features,
    multi_arg=False,
    num_beams=4,
    num_question=1,
    qa_length_penalty=-2.5,
    lower_case=False,
):
    all_pred = collections.OrderedDict()
    all_visualize = collections.OrderedDict()
    all_gold = collections.OrderedDict()
    model.eval()

    for idx, batch in enumerate(eval_dataloader):
        input_ids, input_mask, target_ids, example_indices = batch
        if args.gpu >= 0:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()

        with torch.no_grad():
            if num_beams == 1:
                outputs = model.generate(input_ids=input_ids, attention_mask=input_mask)
            else:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    num_beams=num_beams,
                    early_stopping=False,
                    length_penalty=qa_length_penalty,
                )

        for i, example_index in enumerate(example_indices):
            pre_answer = tokenizer.decode(outputs[i], skip_special_tokens=True)
            if lower_case:
                pre_answer = pre_answer.lower()

            eval_feature = eval_features[example_index.cpu().item()]
            example_id = eval_feature.example_id
            input_sent = eval_examples[example_id].sentence
            if lower_case:
                input_sent = input_sent.lower()

            # for generation model pred
            if example_id not in all_pred:
                all_pred[example_id] = []

            event_type_argument_type = "_".join(
                [eval_feature.event_type, eval_feature.argument_type]
            )

            if not multi_arg:
                if input_sent.find(pre_answer) != -1 and "No Answer" not in pre_answer:
                    all_pred[example_id].append(
                        [
                            event_type_argument_type,
                            input_sent.find(pre_answer),
                            input_sent.find(pre_answer) + len(pre_answer) - 1,
                            pre_answer,
                        ]
                    )

            else:
                if "No Answer" not in pre_answer:
                    pre_arg_list = pre_answer.split(";")
                    cur_start = 0
                    for one_pre_arg in pre_arg_list:
                        one_pre_arg = one_pre_arg.strip()
                        if one_pre_arg == "":
                            continue
                        one_arg_s = input_sent.find(one_pre_arg, cur_start)
                        if one_arg_s == -1:
                            continue
                        one_arg_e = one_arg_s + len(one_pre_arg) - 1
                        cur_start = one_arg_e + 1

                        all_pred[example_id].append(
                            [
                                event_type_argument_type,
                                one_arg_s,
                                one_arg_e,
                                one_pre_arg,
                            ]
                        )

    for example_id, example in enumerate(gold_examples):
        all_gold[example_id] = []
        all_visualize[example_id] = {
            "sentence": example.sentence,
            "gold events": example.events,
        }
        for event in example.events:  #
            event_type = event[0][2]
            for argument in event[1:]:
                argument_start, argument_end, argument_type, argument_str = (
                    argument[0],
                    argument[1],
                    argument[2],
                    argument[3],
                )
                argument_type = loader.normalize_ace_arg(argument_type, event_type)

                event_type_argument_type = "_".join([event_type, argument_type])
                if lower_case:
                    argument_str = argument_str.lower()
                all_gold[example_id].append(
                    [
                        event_type_argument_type,
                        argument_start,
                        argument_end,
                        argument_str,
                    ]
                )

    # get results classification
    gold_arg_n, pred_arg_n, pred_in_gold_n, gold_in_pred_n = 0, 0, 0, 0
    for example_id, _ in enumerate(gold_examples):
        all_visualize[example_id]["predicted arg"] = []
        if example_id not in all_pred:
            all_pred[example_id] = []
        pred_arg = all_pred[example_id]

        all_visualize[example_id]["predicted arg"] = pred_arg

        gold_arg = all_gold[example_id]
        for _ in pred_arg:
            pred_arg_n += 1

        for _ in gold_arg:
            gold_arg_n += 1

        for argument in pred_arg:
            if argument in gold_arg:
                pred_in_gold_n += 1

        for argument in gold_arg:
            if argument in pred_arg:
                gold_in_pred_n += 1

    preds_init = copy.deepcopy(all_visualize)

    if pred_arg_n != 0:
        prec_c = 100.0 * pred_in_gold_n / pred_arg_n
    else:
        prec_c = 0
    if gold_arg_n != 0:
        recall_c = 100.0 * gold_in_pred_n / gold_arg_n
    else:
        recall_c = 0
    if prec_c or recall_c:
        f1_c = 2 * prec_c * recall_c / (prec_c + recall_c)
    else:
        f1_c = 0

    # get results identification
    for example_id, _ in enumerate(gold_examples):
        for argument in all_pred[example_id]:
            argument[0] = argument[0].split("_")[0]
        for argument in all_gold[example_id]:
            argument[0] = argument[0].split("_")[0]

    gold_arg_n, pred_arg_n, pred_in_gold_n, gold_in_pred_n = 0, 0, 0, 0
    for example_id, _ in enumerate(gold_examples):
        pred_arg = all_pred[example_id]
        gold_arg = all_gold[example_id]

        for _ in pred_arg:
            pred_arg_n += 1

        for _ in gold_arg:
            gold_arg_n += 1

        for argument in pred_arg:
            if argument in gold_arg:
                pred_in_gold_n += 1

        for argument in gold_arg:
            if argument in pred_arg:
                gold_in_pred_n += 1

    if pred_arg_n != 0:
        prec_i = 100.0 * pred_in_gold_n / pred_arg_n
    else:
        prec_i = 0
    if gold_arg_n != 0:
        recall_i = 100.0 * gold_in_pred_n / gold_arg_n
    else:
        recall_i = 0
    if prec_i or recall_i:
        f1_i = 2 * prec_i * recall_i / (prec_i + recall_i)
    else:
        f1_i = 0

    result = collections.OrderedDict(
        [
            ("prec_c", prec_c),
            ("recall_c", recall_c),
            ("f1_c", f1_c),
            ("prec_i", prec_i),
            ("recall_i", recall_i),
            ("f1_i", f1_i),
        ]
    )

    return result, preds_init
