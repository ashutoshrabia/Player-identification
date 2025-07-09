# src/feature.py

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

def default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureExtractor:
    """
    Deep CNN-based appearance feature extractor using MobileNetV2 backbone.
    Outputs a 1280-D L2-normalized vector per crop.
    """
    def __init__(self):
        device = default_device()
        # Load pretrained MobileNetV2 and keep only feature layers
        backbone = torch.hub.load(
            'pytorch/vision:v0.13.1', 'mobilenet_v2',
            pretrained=True
        )

        self.model = nn.Sequential(
            *list(backbone.features),
            nn.AdaptiveAvgPool2d(1)
        ).to(device).eval()
        self.device = device

        # Preprocessing
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 256)),         
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225])
        ])

    def __call__(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return torch.zeros(1280, dtype=torch.float32).numpy()

        # Preprocess and forward‑pass
        img = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat_map = self.model(img)           # (1,1280,1,1)
        feat = feat_map.view(-1).cpu()
        feat = feat / (feat.norm(p=2) + 1e-6)    # L2 normalize
        return feat.numpy()
