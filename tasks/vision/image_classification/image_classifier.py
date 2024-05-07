import io
import os
import hashlib
import tempfile
import requests
import numpy as np
from PIL import Image
from argparse import ArgumentParser

import torch
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

from torchvision import models


class image_classifier: 
    def __init__(self,supported_models: list[str] = ['efficientnet-b0', 'efficientnet-b1']):
        self.labels_file_location = "labels.txt"
        self.labels = self._get_image_net_labels()
        self.supported_models = supported_models

    def _fetch(self,url):
        fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode('utf-8')).hexdigest())
        if os.path.isfile(fp) and os.stat(fp).st_size > 0:
            with open(fp, "rb") as f:
                dat = f.read()
        else:
            print("fetching", url)
            dat = requests.get(url).content
            with open(fp+".tmp", "wb") as f:
                f.write(dat)
            os.rename(fp+".tmp", fp)
        return dat

    def _get_image_net_labels(self):
        with open(self.labels_file_location, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip()[10:] for x in f]
        return labels_map

    def classify_image(self,image_url: str, top_n_results: int, model: str):
        try:
            recommendations = []
            if model not in self.supported_models:
                return recommendations, {"status": 0, "error": "Model not supported"}

            self.model = EfficientNet.from_pretrained(model)
            _ = self.model.eval()

            image = Image.open(io.BytesIO(self._fetch(image_url)))  # load any image
            image_transformations = transforms.Compose([
                transforms.Resize(256),                               # resize to a (256,256) image
                transforms.CenterCrop(224),                           # crop centre part of image to get (244, 244) grid
                transforms.ToTensor(),                                # convert to tensor
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),    # normalise image according to imagenet valuess
            ])

            x = image_transformations(image) # [3, H, W]
            x = x.view(1, *x.size())
            x.size()
            with torch.no_grad():
                out = self.model(x)

            probs, indices = torch.topk(out, top_n_results)
            for prob, i in zip(probs[0], indices[0]):
                recommendations.append({"score" : float(prob), "label": self.labels[i]})
            return recommendations , {"status": 1, "error" : None}

        except Exception as e:
            recommendations = []
            return recommendations , {"status": 0, "error": e}
