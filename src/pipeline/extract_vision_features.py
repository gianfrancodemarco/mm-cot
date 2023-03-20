import os
import pandas as pd

from src import constants
from src.pipeline.utils import get_images_paths
from transformers import CLIPVisionModel, DetrForObjectDetection
from src.data.vision_features.transformer_extractor import TransformerExtractor

MODEL_NAME_CLIP = "openai/clip-vit-large-patch14-336"
MODEL_NAME_VIT = "google/vit-large-patch16-224-in21k"
MODEL_NAME_DETR = "facebook/detr-resnet-101-dc5"

models_configs = [{"model_name":MODEL_NAME_CLIP, "model_class": CLIPVisionModel},
          {"model_name":MODEL_NAME_VIT, "model_class": CLIPVisionModel},
          {"model_name":MODEL_NAME_DETR, "model_class": DetrForObjectDetection}]

for model_config in models_configs:

    model_name = model_config["model_name"]
    model_class = model_config["model_class"]
    folder_name = model_name.split("/")[1]

    extractor = TransformerExtractor(model_name=model_name, model_class=model_class)
    base_save_path = os.path.join(constants.FAKEDDIT_VISION_FEATURES_FOLDER_PATH, folder_name, "[split].npy")
    base_data_path = os.path.join(constants.FAKEDDIT_VISION_FEATURES_FOLDER_PATH, "vision_features_[split].csv")

    for split in ["train", "validation", "test"]:

        print(f"\t\t PROCESSING {split}\n\n")

        _data_path = base_data_path.replace("[split]", split)
        _save_path = base_save_path.replace("[split]", split)

        dataframe = pd.read_csv(_data_path)
        images_path = get_images_paths(dataframe)
        extractor.extract_vision_features(file_path=_save_path, list_images_path=images_path)
