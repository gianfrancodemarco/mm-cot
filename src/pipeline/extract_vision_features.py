import os
import pandas as pd

from src import constants
from src.pipeline.utils import get_images_paths
from src.data.vision_features.transformer_extractor import TransformerExtractor

MODEL_NAME_CLIP = "openai/clip-vit-large-patch14-336"
MODEL_NAME_VIT = "google/vit-large-patch16-224-in21k"

extractor = TransformerExtractor(model_name=MODEL_NAME_CLIP)
base_save_path = os.path.join(constants.FAKEDDIT_VISION_FEATURES_FOLDER_PATH, "clip-vit-large-patch14-336") # vit-large-patch16-224-in21k
base_data_path = os.path.join(constants.FAKEDDIT_VISION_FEATURES_FOLDER_PATH, "vision_features_[split].csv")

for split in ["train", "validation", "test"]:

    print(f"\t\t PROCESSING {split}\n\n")

    _data_path = base_data_path.replace("[split]", split)
    dataframe = pd.read_csv(_data_path)
    images_path = get_images_paths(dataframe)
    _save_path = os.path.join(base_save_path, split)
    extractor.extract_vision_features(file_path=_save_path, list_images_path=images_path)

