
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import T5Tokenizer

from src.data.fakeddit.labels import (LabelsTypes, convert_int_to_label,
                                      get_label_column, get_label_text,
                                      get_options_text)

DATASET_PATH = 'data/fakeddit/partial/dataset.csv'
DEFAULT_PROMPT = """Question: Is the source of this information reliable? Context: (Select option A for True, or option B for False) <TEXT> Options: <OPTIONS>"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FakedditDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: T5Tokenizer,
        vision_features: np.ndarray = None,
        rationales: List[str] = None,
        labels_type: LabelsTypes = LabelsTypes.TWO_WAY,
        source_len: int = 512,
        target_len: int = 512,
        image_shape = (100,256)
    ) -> None:

        self.labels_type = labels_type
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.target_len = target_len
        self.vision_features = vision_features
        self.rationales = rationales
        self.image_shape = image_shape

        self.input_ids = torch.tensor([], device=device)
        self.attention_masks = torch.tensor([], device=device)
        self.labels = torch.tensor([], device=device)

        self.image_ids = None
        if self.vision_features is not None:
            self.image_ids = torch.tensor([], device=device)

        self._build_dataset()

    def _build_dataset(self) -> None:

        for index, row in enumerate(self.dataframe.to_dict(orient="records")[200:2000]):

            _rationale = ''
            if self.rationales:
                _rationale = self.rationales[index]

            _input_ids, _attention_mask = self.get_input_ids(
                row["clean_title"], _rationale)

            self.input_ids = torch.cat(
                (self.input_ids, _input_ids.unsqueeze(0)), 0)
            self.attention_masks = torch.cat(
                (self.attention_masks, _attention_mask.unsqueeze(0)), 0)

            self.labels = torch.cat((self.labels, torch.tensor([self.get_label(index)], device=device)), 0)


            if self.vision_features is not None:
                _image_ids = self.get_image_ids(index)
                self.image_ids = torch.cat(
                    (self.image_ids, _image_ids.unsqueeze(0)), 0)

    def get_input_ids(self, title: str, rationale: str) -> Tuple[Tensor, Tensor]:

        full_text = self._get_question_text(title, rationale)
        processed = self.process_data(full_text, self.source_len)

        input_ids = processed["input_ids"].squeeze().to(device)
        attention_mask = processed["attention_mask"].squeeze().to(device)

        return input_ids, attention_mask

    def _get_question_text(self, title: str, rationale: str) -> str:
        options_text = get_options_text(self.labels_type)

        question_text = DEFAULT_PROMPT.replace("<TEXT>", title)
        question_text = question_text.replace("<OPTIONS>", options_text)
        question_text = "\n".join([question_text, rationale])

        return question_text

    def get_image_ids(self, vision_feature_index: int) -> Tensor:

        image_ids = self.vision_features[vision_feature_index]
        if not len(image_ids):
            image_ids = np.zeros(self.image_shape)
        else:
            # TODO: remove on the original data
            image_ids = image_ids[0, :, :]

        return torch.tensor(image_ids).squeeze().to(device)

    def get_label(self, index: str) -> str:
        label_column = get_label_column(self.labels_type)
        return int(self.dataframe.iloc[index][label_column])

    def process_data(
            self,
            text,
            max_length
    ):
        text = " ".join(str(text).split())
        return self.tokenizer.batch_encode_plus(
            [text],
            max_length=max_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.input_ids)

    def __getitem__(self, index) -> dict:

        item = {
            "input_ids": self.input_ids[index].to(torch.long),
            "attention_mask": self.attention_masks[index].to(torch.long),
            "labels":  self.get_label(index),
            "plain_labels": get_label_text(convert_int_to_label(self.get_label(index)))
        }

        if self.image_ids is not None:
            item = {
                **item,
                "image_ids": self.image_ids[index].to(torch.float)
                # "image_ids": torch.zeros(IMG_SHAPE).to(torch.float) FOR EXCLUDE VISION FEATURES
            }

        return item
