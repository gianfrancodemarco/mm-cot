import os

from src.data.pipeline.utils import download_file, unzip

# # Download data
os.system("git clone https://github.com/lupantech/ScienceQA")
os.system("mv ScienceQA/data data")
os.system("rm -r ScienceQA")

# Download vision features
download_file(
    "https://drive.google.com/u/0/uc?id=13B0hc_F_45-UlqPLKSgRz-ALtFQ8kIJr&export=download&confirm=t&uuid=dde4f5c5-a182-41fb-b908-934bb153d5e7&at=ALgDtsyO45szTP40cSRZbVR9h8iU:1677195067586",
    "vision_features.zip")

os.mkdir('vision_features')
unzip('vision_features.zip', '.')

# Download models
download_file(
    'https://drive.google.com/u/0/uc?id=1FtTYOJPHnWnFfCxNC6M3gar4RAX5E21b&export=download&confirm=t&uuid=a5d17c71-252a-421b-9fd0-6ce609da7947&at=ALgDtsxrRDAAnvFetgzK3lXuZ9I_:1676314225641',
    'models.zip'
)

os.mkdir('models')
unzip('models.zip', '.')