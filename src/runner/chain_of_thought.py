import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, T5Tokenizer

from src import constants
from src.constants import PromptFormat, Task
from src.models.t5_multimodal_generation.training_params import (
    get_t5_model, get_training_args)
from src.models.t5_multimodal_generation.utils import (compute_metrics_acc,
                                                       compute_metrics_rougel,
                                                       get_backup_dir,
                                                       get_prediction_filename)
from src.runner.mlflow_logging import MLFlowLogging
from src.runner.runner import Runner
from src.utils import set_random_seed   

class ChainOfThought(Runner):

    def __init__(
        self,
        args
    ):
        self.args = args
        set_random_seed(self.args.seed)
        self.dataframe = None
        self.train_set = None
        self.validation_set = None

        # We keep eval set for compatibility reasons
        self.eval_set = None
        self.test_set = None

        self.t5_model = None
        self.tokenizer = None

        self.save_dir = get_backup_dir(args)
        self.filename = get_prediction_filename(args)

    def set_tokenizer(self, tokenizer: T5Tokenizer):
        self.tokenizer = tokenizer
        return self

    def set_train_set(self, train_set: Dataset):
        self.train_set = train_set
        return self

    def set_validation_set(self, validation_set: Dataset):
        self.validation_set = validation_set
        return self

    def set_eval_set(self, eval_set: Dataset):
        self.eval_set = eval_set
        return self

    def set_test_set(self, test_set: Dataset):
        self.test_set = test_set
        return self

    def set_model(self, model):
        self.model = model
        return self

    def run(self):
        tasks_map = {
            Task.TRAIN.value: self.train,
            Task.EVALUATE.value: self.evaluate,
            Task.INFER.value: self.infer
        }
        task = tasks_map.get(self.args.task)
        task()

    def _get_run_name(self):
        return "_".join([self.filename, self.args.task, self.args.dataset])

    def train(self):
        @MLFlowLogging(experiment_name=self.args.experiment_name, run_name=self._get_run_name())
        def train_mlflow(self):
            self.build_seq2seq_base_trainer(self.train_set, self.eval_set)
            self.seq2seq_trainer.train()
            self.seq2seq_trainer.save_model(self.save_dir)
        return train_mlflow(self)

    def evaluate(self) -> dict:
        @MLFlowLogging(experiment_name=self.args.experiment_name, run_name=self._get_run_name())
        def evaluate_mlflow(self):
            
            """ Generate the textual output for the dataset and returns the metrics """

            output = {
                "metrics": [],
                "predictions": [],
                "targets": []
            }

            for batch in tqdm(DataLoader(dataset=self.test_set, batch_size=self.args.eval_bs, shuffle=False)):

                kwargs = {}
                if getattr(self.test_set, 'image_ids', None) is not None:
                    kwargs['image_ids'] = batch['image_ids']

                out = self.model.generate(
                    batch['input_ids'],
                    **kwargs,
                    repetition_penalty = self.args.repetition_penalty
                )

                prediction = self.tokenizer.batch_decode(
                    out, skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                output["predictions"].extend(prediction)
            # TODO
            # output["targets"] = 
            output["targets"] = [el["plain_labels"] for el in self.test_set]
            output["metrics"] = self._compute_metrics(
                output["predictions"], output["targets"])

            output_prediction_file = os.path.join(
                self.save_dir, f"predictions_{self.filename}_{datetime.now().strftime(constants.DATE_FORMAT)}.json")

            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(output, indent=4))

            return {
                **output["metrics"],
                "task": self.args.task,
                "dataset": self.args.dataset,
                "number_of_examples": len(self.test_set),
                "img_type": self.args.img_type,
                "output": self.args.prompt_format,
                "test_le": self.args.test_le,
                "prompt": self.args.prompt
            }
        return evaluate_mlflow(self)

    def infer(self, sample):
        # Extract EVALUATE common logic in a private method
        pass

    def _compute_metrics(self, predictions, targets):

        metric = compute_metrics_acc
        if self.args.prompt_format == PromptFormat.QUESTION_CONTEXT_OPTIONS_LECTURE_SOLUTION.value:
            metric = compute_metrics_rougel

        return metric(self.tokenizer, predictions, targets)

    def build_seq2seq_base_trainer(self, train_set, eval_set):
        """
            Build a base seq2seq trainer.
            It is mandatory to run this method if t5 model isn't being trained
        """

        print(f"[Model]: Loading {self.args.model}...\n")
        print("[Data]: Reading data...\n")

        self.model = get_t5_model(self.args, self.tokenizer, self.save_dir)

        data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        print("Model parameters: ", self.model.num_parameters())

        self.seq2seq_trainer = Seq2SeqTrainer(
            model=self.model,
            args=get_training_args(self.args, self.save_dir),
            train_dataset=train_set,
            eval_dataset=eval_set,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics_acc if self.args.prompt_format != constants.PromptFormat.QUESTION_CONTEXT_OPTIONS_LECTURE_SOLUTION.value else compute_metrics_rougel
        )

    def _seq2seq_existing_check(self):
        if not self.seq2seq_trainer:
            raise NotImplementedError(
                "ERROR T5000001 | Fit model or if model exists build a seq2seq trainer")
        return True
