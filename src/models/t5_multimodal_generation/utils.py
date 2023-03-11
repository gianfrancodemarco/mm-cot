import os
import re

import evaluate
import nltk
import numpy as np
import torch

from src.constants import PromptFormat


def compute_metrics_rougel(tokenizer, predictions, targets):
    """
    ROUGE-L metric for Rational generation
    """

    metric = evaluate.load("rouge")
    predictions, labels = postprocess_text(
        predictions, targets)

    result = metric.compute(predictions=predictions,
                            references=labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {'rouge-l': result}


def compute_metrics_acc(tokenizer, predictions, targets):
    """
    Accuracy for Answer inference
    """

    correct = 0
    assert len(predictions) == len(targets)
    for idx, pred in enumerate(predictions):
        reference = targets[idx]
        reference = extract_ans(reference)
        extract_pred = extract_ans(pred)
        best_option = extract_pred
        if reference == best_option:
            correct += 1
    return {'accuracy': float(correct) / len(targets)}


def extract_ans(ans):
    pattern = re.compile(r'The answer is \(([A-Z])\)')
    res = pattern.findall(ans)

    if len(res) == 1:
        answer = res[0]  # 'A', 'B', ...
    else:
        answer = "FAILED"

    return answer


def postprocess_text(predictions, labels):
    predictions = [pred.strip() for pred in predictions]
    labels = [label.strip() for label in labels]
    predictions = ["\n".join(nltk.sent_tokenize(pred)) for pred in predictions]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return predictions, labels


def get_backup_dir(args):
    if args.evaluate_dir is not None:
        save_dir = args.evaluate_dir
    else:
        model_name = args.model.replace("/", "-")
        gpu_count = torch.cuda.device_count()
        save_dir = f"{args.output_dir}/{args.user_msg}_{model_name}_{args.img_type}_{args.prompt_format}_lr{args.lr}_bs{args.bs * gpu_count}_op{args.output_len}_ep{args.epoch}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    return save_dir


def get_prediction_filename(args):
    if args.prompt_format == PromptFormat.QUESTION_CONTEXT_OPTIONS_LECTURE_SOLUTION.value:
        return "rationale"
    return "answer"
