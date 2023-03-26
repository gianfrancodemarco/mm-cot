import json
import logging
import os

import dvc.api
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DefaultDataCollator, Trainer

from src import constants
from src.data.fakeddit.dataset import FakedditDataset
from src.data.scienceQA.dataset_img import img_shape
from src.models.baseline_classifiers.model import (TrainingArguments,
                                                   TransformerClassifier,
                                                   TransformerConfig)
from src.models.evaluation.evaluation_metrics import accuracy
from src.runner.mlflow_logging import MLFlowLogging
from dotenv import load_dotenv
from src.utils import set_random_seed

load_dotenv(override=True)
params_show = dvc.api.params_show()['train_evaluate_baseline_classifiers']
set_random_seed(params_show.get("random_state", 42))

img_shape = {
    "detr": (100, 256),
    "clip": (577, 1024),
    "vit": (197, 1024)
}

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')                    # 20 052 034

def get_config():
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dimension=tokenizer.model_max_length,
        input_dim=512,
        hidden_dim=8,
        output_dim=2,
        num_layers=1,
        num_heads=1,
        dropout=0.5,
    )

    img_type = params_show.get('img_type') 
    if img_type:
        assert img_type in img_shape
        config.set_patch_size(img_shape.get(img_type))
        
    return config

def get_datasets():

    VISION_FEATURES_PATHS = {
        "detr": constants.FAKEDDIT_VISION_FEATURES_DETR_PATH,
        "vit": constants.FAKEDDIT_VISION_FEATURES_VIT_PATH,
        "clip": constants.FAKEDDIT_VISION_FEATURES_CLIP
    }
    
    img_type = params_show.get('img_type') 
    use_rationale = params_show.get('use_rationale')

    splits = {}

    for split_name in ['train', 'validation', 'test']:
        logging.info(f"Loading {split_name}")

        dataframe = pd.read_csv(os.path.join(constants.FAKEDDIT_DATASET_PARTIAL_PATH, f"{split_name}.csv"))

        vision_features = None
        if img_type:
            base_path = VISION_FEATURES_PATHS.get(img_type)
            vision_features_path = os.path.join(base_path, f"{split_name}.npy")
            vision_features = np.load(vision_features_path, allow_pickle=True)
    
        rationales = None
        if use_rationale:
            logging.info("Adding rationales to input")
            with open(os.path.join(constants.FAKEDDIT_RATIONALES_DATASET_PATH, f"{split_name}.json"), "r") as f:
                rationale_df = json.loads(f.read())
            rationales = rationale_df['predictions']

        splits[split_name] = FakedditDataset(
            dataframe=dataframe,
            tokenizer=tokenizer,
            vision_features=vision_features, 
            rationales=rationales,
            image_shape=img_shape[img_type] if img_type else None
        )
    return splits["train"], splits["validation"], splits["test"]

config = get_config()
model = TransformerClassifier(config)

training_args = TrainingArguments(
    output_dir=constants.MODEL_PATH,
    evaluation_strategy='steps',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    do_train=True,
    do_eval=True,
    gradient_accumulation_steps=16,
    logging_steps=50,
    eval_steps=50,
    num_train_epochs=100,
    weight_decay=0.15,
    learning_rate=5e-6,
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DefaultDataCollator(),
    compute_metrics=accuracy
)

train_set, validation_set, test_set = get_datasets()

def get_run_name():
   return "_".join([f"{k}_{v}" for (k, v) in params_show.items()])

@MLFlowLogging(experiment_name="baseline", run_name=f"train_{get_run_name()}")
def train():
    trainer.train_dataset, trainer.eval_dataset = train_set, validation_set
    trainer.train()
    return params_show

@MLFlowLogging(experiment_name="baseline", run_name=f"evaluate_{get_run_name()}")
def evaluate():
    results = trainer.evaluate(eval_dataset=test_set, metric_key_prefix="test")
    return {**results, **params_show}

train()
evaluate()