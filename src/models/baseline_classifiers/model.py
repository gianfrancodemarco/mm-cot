
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput


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
        self.set_patch_size(patch_size)

    def set_patch_size(self, patch_size):
        if patch_size:
            self.patch_size = patch_size
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

