import os
import pandas as pd
from src import constants

from src.data.vision_features.transformer_extractor import VisionFeaturesExtractor

def load_data():
    dataframe = pd.read_csv(constants.FAKEDDIT_DATASET_PATH)
    train_data = dataframe[:1200]
    validation_data = dataframe[:200]
    test_data = dataframe[:600]
    return train_data, validation_data, test_data


train_data, validation_data, test_data = load_data()

model_name = "" #nlpconnect/vit-gpt2-image-captioning, 
vision_features_extrctor = VisionFeaturesExtractor(model_name=model_name)

train_data.to_csv(os.path.join(constants.FAKEDDIT_DATASET_PARTIAL_PATH, "vision_features_train.csv"))
test_data.to_csv(os.path.join(constants.FAKEDDIT_DATASET_PARTIAL_PATH, "vision_features_test.csv"))
validation_data.to_csv(os.path.join(constants.FAKEDDIT_DATASET_PARTIAL_PATH, "vision_features_validation.csv"))
