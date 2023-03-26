import os
from enum import Enum
from pathlib import Path

DATE_FORMAT = '%H_%M_%S'
ROOT_PATH = Path(__file__).parent.parent
SRC_PATH = os.path.join(ROOT_PATH, "src")
DATA_PATH = os.path.join(ROOT_PATH, "data")
MODEL_PATH = os.path.join(ROOT_PATH, "models")

SCIENCEQA_VISION_FEATURES_PATH = os.path.join(DATA_PATH, "vision_features")
SCIENCEQA_DATASET_PATH = os.path.join(DATA_PATH, "dataset", "scienceqa")
SCIENCEQA_PROBLEMS_PATH = os.path.join(SCIENCEQA_DATASET_PATH, "problems.json")
SCIENCEQA_PID_SPLITS = os.path.join(SCIENCEQA_DATASET_PATH, "pid_splits.json")
SCIENCEQA_NAME_MAP = os.path.join(SCIENCEQA_VISION_FEATURES_PATH, "name_map.json")
SCIENCEQA_RESNET = os.path.join(SCIENCEQA_VISION_FEATURES_PATH, "resnet.npy")
SCIENCEQA_CLIP = os.path.join(SCIENCEQA_VISION_FEATURES_PATH, "clip.npy")
SCIENCEQA_DETR = os.path.join(SCIENCEQA_VISION_FEATURES_PATH, "detr.npy")

FAKEDDIT_DATASET_PARTIAL_PATH = os.path.join(DATA_PATH, "fakeddit", "partial")
FAKEDDIT_DATASET_PATH = os.path.join(FAKEDDIT_DATASET_PARTIAL_PATH, "dataset.csv")
FAKEDDIT_IMG_DATASET_PATH = os.path.join(DATA_PATH, "fakeddit", "images")
FAKEDDIT_RATIONALES_DATASET_PATH = os.path.join(FAKEDDIT_DATASET_PARTIAL_PATH, "rationales")

FAKEDDIT_VISION_FEATURES_FOLDER_PATH = os.path.join(FAKEDDIT_DATASET_PARTIAL_PATH, "vision_features")
FAKEDDIT_VISION_FEATURES_DETR_SUB_PATH = os.path.join(FAKEDDIT_VISION_FEATURES_FOLDER_PATH, "vision_features_600.npy")
FAKEDDIT_VISION_FEATURES_DETR_FULL_PATH = os.path.join(FAKEDDIT_VISION_FEATURES_FOLDER_PATH, "vision_features.npy")

FAKEDDIT_VISION_FEATURES_DETR_PATH = os.path.join(FAKEDDIT_VISION_FEATURES_FOLDER_PATH, "detr-resnet-101-dc5")
FAKEDDIT_VISION_FEATURES_VIT_PATH = os.path.join(FAKEDDIT_VISION_FEATURES_FOLDER_PATH, "vit-large-patch16-224-in21k")
FAKEDDIT_VISION_FEATURES_CLIP = os.path.join(FAKEDDIT_VISION_FEATURES_FOLDER_PATH, "clip-vit-large-patch14-336")

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


class DatasetType(Enum):
    FAKEDDIT = "FAKEDDIT"
    SCIENCEQA = "SCIENCEQA"

class ModelOutput(Enum):
    RATIONALE = "rationale"
    ANSWER = "answer"