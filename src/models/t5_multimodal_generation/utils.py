import os
import re

import nltk
import torch
import numpy as np
from src.constants import PromptFormat


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
