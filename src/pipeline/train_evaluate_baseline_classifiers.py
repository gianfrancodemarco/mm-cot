
import json
import os

import dvc.api
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, DefaultDataCollator, Trainer

from src import constants
from src.data.fakeddit.dataset import FakedditDataset
from src.data.scienceQA.dataset_img import img_shape
from src.models.baseline_classifiers.model import (TrainingArguments,
                                                   TransformerClassifier,
                                                   TransformerConfig)
from src.models.evaluation.evaluation_metrics import accuracy

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# FOR DEBUG ONLY, REMOVE THIS
#params_show = dvc.api.params_show(stages='train_evaluate_baseline_classifiers')
params_show = {
    "img_type": "detr",
    "use_rationale": True
}

img_shape = {
    "detr": (100, 256)
}

def get_config():
    config = TransformerConfig(
        input_dim=512,
        hidden_dim=512,
        output_dim=2,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
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

        dataframe = pd.read_csv(os.path.join(constants.FAKEDDIT_DATASET_PARTIAL_PATH, f"{split_name}.csv"))

        vision_features = None
        if img_type:
            base_path = VISION_FEATURES_PATHS.get(img_type)
            vision_features_path = os.path.join(base_path, f"{split_name}.npy")
            vision_features = np.load(vision_features_path, allow_pickle=True)
    
        rationales = None
        if use_rationale:
            with open(os.path.join(constants.FAKEDDIT_RATIONALES_DATASET_PATH, f"{split_name}.json"), "r") as f:
                rationale_df = json.loads(f.read())
            rationales = rationale_df['predictions']

        splits[split_name] = FakedditDataset(
            dataframe=dataframe,
            tokenizer=tokenizer,
            vision_features=vision_features, 
            rationales=rationales
        )

    #train_loader = DataLoader(splits["train"], batch_size=4, shuffle=True)
    #validation_loader = DataLoader(splits["validation"], batch_size=4, shuffle=False)
    #test_loader = DataLoader(splits["test"], batch_size=4, shuffle=False)

    return splits["train"], splits["validation"], splits["test"]

config = get_config()
model = TransformerClassifier(config)

training_args = TrainingArguments(
    output_dir="./",
    evaluation_strategy='steps',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    do_train=True,
    do_eval=True,
    gradient_accumulation_steps=16,
    logging_steps=50,
    eval_steps=50,
    overwrite_output_dir=True,
    save_total_limit=1,
    num_train_epochs=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DefaultDataCollator(),
    compute_metrics=accuracy
)

train_set, validation_set, test_set = get_datasets()

def train():
    trainer.train_dataset, trainer.eval_dataset = train_set, validation_set
    trainer.train()

def evaluate():
    trainer.evaluate(eval_dataset=test_set)

train()
evaluate()