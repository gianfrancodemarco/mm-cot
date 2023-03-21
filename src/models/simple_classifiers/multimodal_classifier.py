
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_metric
from torch.utils.data import DataLoader
from transformers import (BertTokenizer, DefaultDataCollator, Trainer,
                          TrainingArguments)
from transformers.modeling_outputs import SequenceClassifierOutput

from src import constants
from src.data.fakeddit.dataset import FakedditDataset


class TransformerConfig:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        patch_size: Tuple[int, int] = None
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.patch_num, self.patch_dim = None, None
        if patch_size:
            self.patch_num, self.patch_dim = patch_size


class TransformerClassifier(nn.Module):
    def __init__(self, config):
        super(TransformerClassifier, self).__init__()

        self.text_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.input_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.text_encoder_layer,
            num_layers=config.num_layers)

        if config.patch_dim:
            self.vision_fc = nn.Linear(config.patch_num * config.patch_dim, config.input_dim)
            self.fc = nn.Linear(2*config.input_dim, config.output_dim)
        else:
            self.fc = nn.Linear(config.input_dim, config.output_dim)

        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_ids: Optional[torch.FloatTensor] = None,
        labels: Optional[int] = None
    ):

        input_ids = self.transformer_encoder(input_ids.to(torch.float32))
        if image_ids is not None:
            image_ids = image_ids.view(image_ids.shape[0], -1)
            image_ids = self.vision_fc(image_ids)
            combined_features = torch.cat((input_ids, image_ids), dim=1)
        else:
            combined_features = input_ids

        logits = self.fc(combined_features)

        return SequenceClassifierOutput(
            loss=self.loss(logits, labels),
            logits=logits
        )


# Definisci i parametri di configurazione del modello
config = TransformerConfig(
    input_dim=512,
    hidden_dim=512,
    output_dim=2,
    num_layers=4,
    num_heads=8,
    dropout=0.1,
    #patch_size=(100, 256)
)

# Crea un'istanza del modello con i parametri di configurazione
model = TransformerClassifier(config)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_set = FakedditDataset(
    dataframe=pd.read_csv(constants.FAKEDDIT_TRAIN_SET_PATH),
    tokenizer=tokenizer,
    #vision_features=np.load(constants.FAKEDDIT_VISION_FEATURES_DETR_TRAIN, allow_pickle=True)
)
validation_set = FakedditDataset(
    dataframe=pd.read_csv(constants.FAKEDDIT_VALIDATION_SET_PATH),
    tokenizer=tokenizer,
    #vision_features=np.load(constants.FAKEDDIT_VISION_FEATURES_DETR_VALIDATION, allow_pickle=True)
)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = DataLoader(train_set, batch_size=4, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=4, shuffle=False)

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


metric = load_metric('accuracy')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=validation_set,
    data_collator=DefaultDataCollator(),
    compute_metrics=compute_metrics
)

trainer.train()
