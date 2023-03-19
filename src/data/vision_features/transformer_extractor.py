import os
import sys
from transformers import ViTImageProcessor, ViTModel
from PIL import Image, UnidentifiedImageError
import pandas as pd
import torch
import numpy as np
import dvc.api
from dvc.exceptions import DvcException


class TransformerExtractor:
    
    def __init__(self, model_name: str = None):
        model_name = model_name if model_name else self._get_model_name()
        self.image_processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name) #change model loader
        
    def _get_model_name(self):

        try:
            params = dvc.api.params_show()
            model_name = params['feature_extraction']['model-name']
        except DvcException:
            model_name = "facebook/detr-resnet-101-dc5"

        return model_name

    def extract_vision_features(self, list_images_path: list, file_path:str):
        
        vision_features = self.load_checkpoint(file_path)
        checkpoint = len(vision_features)

        with torch.no_grad():
            for index, image_path in enumerate(list_images_path):
                vision_feature = np.array([])
                print(f"PROCESSING #{index + 1}: {image_path}")

                try:
                    image = Image.open(image_path)
                    inputs = self.image_processor(images=image, return_tensors="pt")
                    outputs = self.model(**inputs) 

                    # the last hidden states are the final query embeddings of the Transformer decoder
                    # these are of shape (batch_size, num_queries, hidden_size)
                    vision_feature = outputs.last_hidden_state.numpy()

                except (FileNotFoundError,  ValueError, UnidentifiedImageError) as err:
                    print(f"{image_path} || {err}")
                    
                vision_features.append(vision_feature)
                np.save(file_path, np.asarray(vision_features))

    def load_checkpoint(self, checkpoint_path: str):
        vision_features = []

        if not checkpoint_path:
            return vision_features

        try:
            vision_features = np.load(checkpoint_path, allow_pickle=True).tolist()
        except (FileNotFoundError, PermissionError):
            print("Checkpoint not found")
        
        return vision_features

"""
vision_features_extrctor = TransformerExtractor()
save_path = os.path.join(constants.FAKEDDIT_DATASET_PARTIAL_PATH, "vision_features.npy")
checkpoint = len(vision_features_extrctor.load_checkpoint(save_path))
dataframe = pd.read_csv(constants.FAKEDDIT_DATASET_PATH)
list_img_path = [ os.path.join(constants.FAKEDDIT_IMG_DATASET_PATH, f"{row['id']}.jpg") for row in dataframe.to_dict(orient="records")[checkpoint:]]

vision_features_extrctor.extract_vision_features(list_img_path, save_path)
"""