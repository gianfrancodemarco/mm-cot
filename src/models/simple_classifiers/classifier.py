
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer

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

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x


# Definisci i parametri di configurazione del modello
config = TransformerConfig(input_dim=512, hidden_dim=2048,
                           output_dim=2, num_layers=6, num_heads=8, dropout=0.1)

# Crea un'istanza del modello con i parametri di configurazione
model = TransformerClassifier(config)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataframe = pd.read_csv(constants.FAKEDDIT_DATASET_PATH)
test_set = FakedditDataset(
    dataframe=dataframe,
    tokenizer=tokenizer,
)

tokenized_texts = torch.tensor(test_set[0]['input_ids']).to(torch.float32)
tokenized_texts = torch.stack((tokenized_texts, tokenized_texts))
outputs = model(tokenized_texts)
_, predicted = torch.max(outputs, dim=1)
print(predicted.tolist())
