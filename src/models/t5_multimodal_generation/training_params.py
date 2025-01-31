import torch
from transformers import Seq2SeqTrainingArguments, T5Tokenizer

from src.data.scienceQA.dataset_img import ScienceQADatasetImg, img_shape
from src.data.scienceQA.dataset_std import ScienceQADatasetStd
from src.models.t5_multimodal_generation.model import (
    T5ForConditionalGeneration, T5ForMultimodalGeneration)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_t5_model(args, tokenizer: T5Tokenizer, save_dir: str):
    if is_img_type_known(args):
        padding_idx = tokenizer._convert_token_to_id(tokenizer.pad_token)
        patch_size = img_shape[args.img_type]
        model = T5ForMultimodalGeneration.from_pretrained(
            args.model, patch_size=patch_size, padding_idx=padding_idx, save_dir=save_dir)
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.model)
    model.to(device=device)
    return model


def get_training_data(args, dataframe, tokenizer):

    problems = dataframe['problems']
    qids = dataframe['qids']
    train_qids = qids['train']
    test_qids = qids['test']
    val_qids = qids['val']

    if is_img_type_known(args):
        name_maps = dataframe['name_maps']
        image_features = dataframe['image_features']

        train_set = None
        # train_set = ScienceQADatasetImg(
        #     problems,
        #     train_qids,
        #     tokenizer,
        #     args.input_len,
        #     args.output_len,
        #     args,
        #     image_features=image_features,
        #     name_maps=name_maps
        # )
        eval_set = ScienceQADatasetImg(
            problems,
            val_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features=image_features,
            test_le=args.eval_le,
            name_maps=name_maps
        )
        test_set = ScienceQADatasetImg(
            problems,
            test_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features=image_features,
            test_le=args.test_le,
            name_maps=name_maps
        )
    else:
        train_set = ScienceQADatasetStd(
            problems,
            train_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
        )
        eval_set = ScienceQADatasetStd(
            problems,
            val_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.eval_le,
        )

        test_set = ScienceQADatasetStd(
            problems,
            test_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.test_le,
        )

    return train_set, eval_set, test_set


def get_training_args(args, output_dir):

    shared_params = {
        "output_dir": output_dir,
        "do_train": True if args.evaluate_dir is None else False,
        "logging_strategy": "steps",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "learning_rate": args.lr,
        "eval_accumulation_steps": args.eval_acc,
        "per_device_train_batch_size": args.bs,
        "per_device_eval_batch_size": args.eval_bs,
        "weight_decay": 0.01,
        "num_train_epochs": args.epoch,
        "predict_with_generate": args.use_generate,
        "report_to": "none",
        "save_total_limit": 1,
        "resume_from_checkpoint": True
    }

    # only use the last model for evaluation to save time
    if args.final_eval:
        params = {
            **shared_params,
            "do_eval": False,
            "evaluation_strategy": "no"
        }
    # evaluate at each epoch
    else:
        params = {
            **shared_params,
            "do_eval": True,
            "evaluation_strategy": "epoch",
            "metric_for_best_model": "accuracy" if args.prompt_format != "QCM-LE" else "rougeL",
            "load_best_model_at_end": True,
        }

    return Seq2SeqTrainingArguments(**params)


def is_img_type_known(args):
    img_type = args.img_type
    return img_type in ['facebook_detr', 'cooelf_detr', 'clip', 'resnet']
