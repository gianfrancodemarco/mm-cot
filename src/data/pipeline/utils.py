import os
import zipfile

import requests
from tqdm import tqdm

from src import constants


def unzip(file, dest):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(dest)


def download_file(url, file_name):
    response = requests.get(url, stream=True)
    print(response)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte
    tqdm_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(file_name, 'wb') as f:
        for data in response.iter_content(block_size):
            f.write(data)
            tqdm_bar.update(len(data))
    tqdm_bar.close()


def get_images_paths(dataframe: pd.DataFrame) -> list:
    images_paths = []
    base_path = constants.FAKEDDIT_IMG_DATASET_PATH
    for row in dataframe.iterrows():
        image_path = os.path.join(base_path, row['id'])
        images_paths.append(image_path)
    return images_paths