import json
import os
import random

import evaluate
import numpy as np
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer
from transformers import T5Tokenizer
from torch.utils.data import Dataset

from datetime import datetime
from src.data.science_qa_dataset_iterator import ScienceQADatasetIterator
from src.models.evaluation.evaluation import get_scores
from src.models.t5_multimodal_generation.training_params import get_t5_model, get_training_args
from src.models.t5_multimodal_generation.utils import extract_predictions_and_targets, extract_ans, \
    postprocess_text
from src.models.t5_multimodal_generation.utils import make_backup_dir
from src import constants


class T5ForMultimodalGenerationService:
    seq2seq_trainer = None

    def __init__(self, dataframe, args, tokenizer):
        self.args = args
        self.dataframe = dataframe
        self.save_dir = make_backup_dir(args)
        self.tokenizer = tokenizer or T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=self.args.model)

    def fit(self, train_set, eval_set):
        self.build_seq2seq_base_trainer(train_set, eval_set)
        self.seq2seq_trainer.train()
        self.seq2seq_trainer.save_model(self.save_dir)

    def build_seq2seq_base_trainer(self, train_set, eval_set):
        """
            Build a base seq2seq trainer.
            It is mandatory to run this method if t5 model isn't being trained
        """

        print(f"[Model]: Loading {self.args.model}...\n")
        print("[Data]: Reading data...\n")

        model = get_t5_model(self.args, self.tokenizer, self.save_dir)
        self.model = model

        data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        print("Model parameters: ", model.num_parameters())

        training_args = get_training_args(self.args, self.save_dir)

        self.seq2seq_trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=eval_set,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_acc if self.args.prompt_format != constants.PromptFormat.QUESTION_CONTEXT_OPTIONS_LECTURE_SOLUTION.value else self.compute_metrics_rougel
        )

    def evaluate(self, dataset: Dataset):
        """ Generate the textual output for the dataset and computes the metrics """

        self._seq2seq_existing_check()  # TODO REMOVE

        output = {
            "metrics": [],
            "predictions": [],
            "targets": []
        }

        for elem in dataset:
            
            out = self.model.generate(
                elem['input_ids'][None, :],
                image_ids=elem['image_ids'][None, :],
            )

            prediction = self.tokenizer.batch_decode(
                out, skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            
            output["predictions"].extend(prediction)
            output["targets"].append(elem["labels"])

        output_prediction_file = os.path.join(
            self.save_dir, f"predictions_ans_test_{datetime.now().strftime('%H_%M_%S')}.json")

        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(output, indent=4))

    def inference(self, data):
        """Generate the rationale for the input data"""

        self._seq2seq_existing_check()

        preds = []

        for batch in ScienceQADatasetIterator(dataset=data, batch_size=1):
            self.model.config.max_length = 512
            self.model.config.repetition_penalty = 10.0
            self.model.config.length_penalty = 10.0

            out = self.model.generate(
                batch[0]['input_ids'][None, :],
                image_ids=batch[0]['image_ids'][None, :],
            )

            predictions = self.tokenizer.batch_decode(
                out, skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            predictions = [pred.strip() for pred in predictions]

            preds.extend(predictions)

        output_prediction_file = os.path.join(
            self.save_dir, f"predictions_rel_eval_{datetime.now().strftime('%H_%M_%S')}.json")

        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(preds, indent=4))

    def _seq2seq_existing_check(self):
        if not self.seq2seq_trainer:
            raise NotImplementedError(
                "ERROR T5000001 | Fit model or if model exists build a seq2seq trainer")
        return True

    def compute_metrics_rougel(self, output):

        predictions, label_ids = output

        predictions, targets = extract_predictions_and_targets(
            predictions, label_ids, self.tokenizer)

        metric = evaluate.load("rouge")
        decoded_predictions, decoded_labels = postprocess_text(
            predictions, targets)

        result = metric.compute(predictions=decoded_predictions,
                                references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(
            pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    def compute_metrics_acc(self, output):
        """
        Accuracy for answer inference
        """

        predictions, label_ids = output

        predictions, targets = extract_predictions_and_targets(
            predictions, label_ids, self.tokenizer)
        correct = 0
        assert len(predictions) == len(targets)
        for idx, pred in enumerate(predictions):
            reference = targets[idx]
            reference = extract_ans(reference)
            extract_pred = extract_ans(pred)
            best_option = extract_pred
            if reference == best_option:
                correct += 1
        return {'Accuracy': 1.0 * correct / len(targets)}
