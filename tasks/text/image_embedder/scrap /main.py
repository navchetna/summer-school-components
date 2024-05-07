import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np

class image_embedder:
    def __init__(self):
        self.model = resnet50(pretrained=True)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def embed_image(self, img_path):
        img = Image.open(img_path)
        img = self.transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            feat = self.model.forward(img).squeeze()

        feat = feat.cpu().numpy()

        return feat

    def calculate_similarity(self, first_embedding, second_embedding):
        
        dot_product = np.dot(first_embedding, second_embedding)
        norm_v1 = np.linalg.norm(first_embedding)
        norm_v2 = np.linalg.norm(second_embedding)
        cosine_similarity = dot_product / (norm_v1 * norm_v2)
        
        return cosine_similarity