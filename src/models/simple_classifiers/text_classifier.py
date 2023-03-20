
from typing import Optional

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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout


class TransformerClassifier(nn.Module):
    def __init__(self, config):
        super(TransformerClassifier, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.input_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=config.num_layers)

        self.fc = nn.Linear(config.input_dim, config.output_dim)
        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[int] = None
    ):
        input_ids = self.transformer_encoder(input_ids.to(torch.float32))
        input_ids = self.fc(input_ids)

        return SequenceClassifierOutput(
            loss=self.loss(input_ids, labels),
            logits=input_ids
        )


# Definisci i parametri di configurazione del modello
config = TransformerConfig(
    input_dim=512,
    hidden_dim=512,
    output_dim=2,
    num_layers=4,
    num_heads=8,
    dropout=0.1
)

# Crea un'istanza del modello con i parametri di configurazione
model = TransformerClassifier(config)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_set = FakedditDataset(
    dataframe=pd.read_csv(constants.FAKEDDIT_TRAIN_SET_PATH),
    tokenizer=tokenizer,
)
validation_set = FakedditDataset(
    dataframe=pd.read_csv(constants.FAKEDDIT_VALIDATION_SET_PATH),
    tokenizer=tokenizer,
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
    save_total_limit=10,
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
