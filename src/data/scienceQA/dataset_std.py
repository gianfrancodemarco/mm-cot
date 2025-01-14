import json

import torch
from torch.utils.data import Dataset

from src.models.prompt import build_train_pair


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ScienceQADatasetStd(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
            self,
            problems,
            qids,
            tokenizer,
            source_len,
            target_len,
            args,
            test_le=None
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """

        self.tokenizer = tokenizer
        self.data = {qid: problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len

        self.source_ids = torch.tensor([]).to(device)
        self.source_masks = torch.tensor([]).to(device)
        self.target_ids = torch.tensor([]).to(device)
        self.plain_targets = []

        if test_le is not None:
            test_le_data = json.load(open(test_le))["preds"]
        else:
            test_le_data = None

        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            prompt, target = build_train_pair(
                problems, qid, args, curr_le_data)

            # SOURCE
            source = self.process_data(prompt, self.source_len)
            source_id = source["input_ids"].squeeze().to(device)
            source_mask = source["attention_mask"].squeeze().to(device)

            # TARGET
            self.plain_targets.append(target)
            target = self.process_data(target, self.summ_len)
            target = target["input_ids"].squeeze().to(device)

            self.source_ids = torch.cat(
                (self.source_ids, source_id.unsqueeze(0)), 0)
            self.source_masks = torch.cat(
                (self.source_masks, source_mask.unsqueeze(0)), 0)
            self.target_ids = torch.cat(
                (self.target_ids, target.unsqueeze(0)), 0)

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.target_ids)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        return {
            "input_ids": self.source_ids[index].to(torch.long),
            "attention_mask": self.source_masks[index].to(torch.long),
            "labels": self.target_ids[index].to(torch.long).tolist(),
            "plain_labels": self.plain_targets[index] 
        }

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
