import os
import pandas as pd
from src import constants

def load_data():
    dataframe = pd.read_csv(constants.FAKEDDIT_DATASET_PATH)
    train_data = dataframe[:1200]
    validation_data = dataframe[1200:1400]
    test_data = dataframe[1400:2000]
    return train_data, validation_data, test_data


train_data, validation_data, test_data = load_data()

train_data.to_csv(os.path.join(constants.FAKEDDIT_VISION_FEATURES_FOLDER_PATH, "vision_features_train.csv"))
test_data.to_csv(os.path.join(constants.FAKEDDIT_VISION_FEATURES_FOLDER_PATH, "vision_features_test.csv"))
validation_data.to_csv(os.path.join(constants.FAKEDDIT_VISION_FEATURES_FOLDER_PATH, "vision_features_validation.csv"))
