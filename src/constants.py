from enum import Enum

import os
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
SRC_PATH = os.path.join(ROOT_PATH, "src")
DATA_PATH = os.path.join(ROOT_PATH, "data")

SCIENCEQA_VISION_FEATURES_PATH = os.path.join(DATA_PATH, "vision_features")
SCIENCEQA_DATASET_PATH = os.path.join(DATA_PATH, "dataset", "scienceqa")
SCIENCEQA_PROBLEMS_PATH = os.path.join(SCIENCEQA_DATASET_PATH, "problems.json")
SCIENCEQA_PID_SPLITS = os.path.join(SCIENCEQA_DATASET_PATH, "pid_splits.json")
SCIENCEQA_NAME_MAP = os.path.join(SCIENCEQA_VISION_FEATURES_PATH, "name_map.json")
SCIENCEQA_RESNET = os.path.join(SCIENCEQA_VISION_FEATURES_PATH, "resnet.npy")
SCIENCEQA_CLIP = os.path.join(SCIENCEQA_VISION_FEATURES_PATH, "clip.npy")
SCIENCEQA_DETR = os.path.join(SCIENCEQA_VISION_FEATURES_PATH, "detr.npy")

FAKEDDIT_DATASET_PATH = os.path.join(DATA_PATH, "fakeddit", "partial", "dataset.csv")
FAKEDDIT_IMG_DATASET_PATH = os.path.join(DATA_PATH, "fakeddit", "images")
FAKEDDIT_VISION_FEATURES_PATH = os.path.join(DATA_PATH, "fakeddit", "partial", "vision_features.npy")

class PromptFormat(Enum):
    """
    Possible values for the prompt format
    The template is:
    <INPUT_FORMAT>-<OUTPUT_FORMAT>
    """

    QUESTION_CONTEXT_OPTIONS_ANSWER = "QCM-A"
    QUESTION_CONTEXT_OPTIONS_LECTURE_SOLUTION = "QCM-LE"
    QUESTION_CONTEXT_OPTIONS_SOLUTION_ANSWER = "QCMG-A"  # Does G stand for solution?
    QUESTION_CONTEXT_OPTIONS_LECTURE_SOLUTION_ANSWER = "QCM-LEA"
    QUESTION_CONTEXT_OPTIONS_ANSWER_LECTURE_SOLUTION = "QCM-ALE"

    @classmethod
    def get_values(cls):
        return [e.value for e in cls]


class Task(Enum):
    """Possible model action"""
    EVALUATE = "EVALUATE"
    TRAIN = "TRAIN"
    INFER = "INFER"