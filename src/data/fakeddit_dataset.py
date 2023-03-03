
import pandas as pd
import torch
from torch.utils.data import IterableDataset
from torch import Tensor
# test_le: Probably it is the previously generated rationale, needed to inference the answer (so it will be null when)
# inferencing the rationale

# prompt: The question
#   e.i: "Question: What does the verbal irony in this text suggest?\nAccording to Mr. Herrera's kids, his snoring is as quiet as a jackhammer.\nContext: N/A\nOptions: (A) The snoring is loud. (B) The snoring occurs in bursts.\nSolution:"
# target: The rationale
#   e.i: "Solution: Figures of speech are words or phrases that use language in a nonliteral or unusual way. They can make writing more expressive.\\nVerbal irony involves saying one thing but implying something very different. People often use verbal irony when they are being sarcastic.\\nOlivia seems thrilled that her car keeps breaking down.\\nEach breakdown is as enjoyable as a punch to the face. The text uses verbal irony, which involves saying one thing but implying something very different.\\nAs quiet as a jackhammer suggests that the snoring is loud. A jackhammer is not quiet, and neither is Mr. Herrera's snoring.."

# source = self.process_data(prompt, self.source_len)
# source_id = source["input_ids"].squeeze().to(device)
# source_mask = source["attention_mask"].squeeze().to(device)
# target = target["input_ids"].squeeze().to(device)


# "input_ids": self.source_ids[index].to(torch.long),
# "attention_mask": self.source_masks[index].to(torch.long),
# "image_ids": self.image_ids[index].to(torch.float),
# "labels": self.target_ids[index].to(torch.long).tolist(),

DATASET_PATH = 'data/fakeddit/partial/dataset.csv'


def load_data(args):
    return pd.read_csv(DATASET_PATH)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FakedditDataset(IterableDataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        images_path: str,
        tokenizer: any,
        max_length: int = 512
    ) -> None:

        self.input_ids = torch.tensor([], device=device)
        self.image_ids = torch.tensor([], device=device)
        self.attention_masks = torch.tensor([], devic=device)
        self.labels = torch.tensor([], device=device)

        for row in dataframe.to_dict(orient="records"):
            _input_ids = self.get_input_ids(row)
            _image_ids = self.get_image_ids(row)
            _attention_mask = self.get_attention_mask(row)
            _labels = self.get_labels(row)

            self.input_ids = torch.cat(
                (self.input_ids, _input_ids.unsqueeze(0)), 0)
            self.image_ids = torch.cat(
                (self.image_ids, _image_ids.unsqueeze(0)), 0)
            self.attention_masks = torch.cat(
                (self.attention_masks, _attention_mask.unsqueeze(0)), 0)
            self.labels = torch.cat(
                (self.labels, _labels.unsqueeze(0)), 0)

    def get_input_ids(self, row: dict) -> Tensor:
        pass

    def get_image_ids(self, row: dict) -> Tensor:
        pass

    def get_attention_mask(self, row: dict) -> Tensor:
        pass

    def get_labels(self, row: dict) -> Tensor:
        pass

    def process_data(
            self,
            text
    ):
        text = " ".join(str(text).split())
        return self.tokenizer.batch_encode_plus(
            [text],
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def __getitem__(self, index) -> dict:

        return {
            "input_ids": self.input_ids[index].to(torch.long),
            "attention_mask": self.attention_masks[index].to(torch.long),
            "labels": self.labels[index].to(torch.long).tolist(),
        }
