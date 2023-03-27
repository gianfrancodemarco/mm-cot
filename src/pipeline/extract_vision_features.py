import os

import dvc.api
import numpy as np
import pandas as pd
from transformers import CLIPVisionModel, DetrForObjectDetection

from src import constants
from src.data.vision_features.detr_extractor import DetrExtractor
from src.data.vision_features.transformer_extractor import TransformerExtractor
from src.pipeline.utils import get_images_paths

params = dvc.api.params_show()['feature_extraction']
model_name = params['model_name']

MODEL_NAME_CLIP = "openai/clip-vit-large-patch14-336"
MODEL_NAME_VIT = "google/vit-large-patch16-224-in21k"
MODEL_NAME_DETR = "facebook/detr-resnet-101-dc5"
MODEL_NAME_DETR_MMCOT = "cooelf/detr_resnet101_dc5"

TRANSFORMER_EXTRACTORS = [MODEL_NAME_CLIP, MODEL_NAME_VIT, MODEL_NAME_DETR]

models_configs = {
    MODEL_NAME_CLIP: CLIPVisionModel,
    MODEL_NAME_VIT: CLIPVisionModel,
    MODEL_NAME_DETR: DetrForObjectDetection
}

if model_name in TRANSFORMER_EXTRACTORS:
    model_class = models_configs[model_name]
    extractor = TransformerExtractor(model_name = model_name, model_class = model_class)
else:
    extractor=DetrExtractor()

folder_name=model_name.replace("/", '_')

base_save_path=os.path.join(
    constants.FAKEDDIT_VISION_FEATURES_FOLDER_PATH, folder_name)
base_data_path=os.path.join(constants.FAKEDDIT_DATASET_PARTIAL_PATH)

if not os.path.exists(base_save_path):
    os.makedirs(base_save_path)

for split in ["train", "validation", "test"]:

    print(f"\t\t PROCESSING {split}\n\n")

    _save_path=os.path.join(base_save_path, f"{split}.npy")
    _data_path=os.path.join(base_data_path, f"{split}.csv")

    dataframe=pd.read_csv(_data_path)
    images_path=get_images_paths(dataframe)
    extractor.extract_vision_features(file_path = _save_path, list_images_path =images_path)

whole_features=[]
for split in ["train", "validation", "test"]:
    print(f"Processing {split}")
    _save_path=os.path.join(base_save_path, f"{split}.npy")
    whole_features.append(np.load(_save_path, allow_pickle=True))
whole_features= np.concatenate(whole_features, axis = 0)
_save_path=os.path.join(base_save_path, "dataset.npy")
np.save(_save_path, np.asarray(whole_features))
