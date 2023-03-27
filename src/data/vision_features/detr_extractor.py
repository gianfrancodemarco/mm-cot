import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


class DetrExtractor:

    def __init__(self) -> None:
        self.detr_model = torch.hub.load(
            'cooelf/detr', 'detr_resnet101_dc5', pretrained=True)
        self.detr_model.eval()

    def extract_vision_features(self, list_images_path: list, file_path: str):

        vision_features = []

        with torch.no_grad():
            for image_path in tqdm(list_images_path):
                vision_feature = np.array([])
                try:
                    transform = T.Compose([
                        T.Resize(224),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
                    ])
                    with torch.no_grad():
                        img = Image.open(image_path).convert("RGB")
                        input = transform(img).unsqueeze(0)
                        vision_feature = self.detr_model(input)[-1].numpy()

                except (FileNotFoundError,  ValueError, UnidentifiedImageError) as err:
                    print(f"{image_path} || {err}")

                vision_features.append(vision_feature)
                np.save(file_path, np.asarray(vision_features))