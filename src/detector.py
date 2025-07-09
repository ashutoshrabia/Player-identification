import torch
from ultralytics import YOLO
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential, ModuleList, ModuleDict
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU, ReLU
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample
from ultralytics.nn.modules.conv import Conv, Concat
from ultralytics.nn.modules.block import C1, C2, C3, C2f, Bottleneck, SPPF, DFL
from ultralytics.nn.modules.head import Detect

class YoloDetector:
    
    def __init__(
        self,
        weight_path: str = "weights/best.pt",
        conf: float = 0.25,
        classes: list[int] | None = None,
        device: str | None = None
    ):
        # Allowlist Ultralytics and PyTorch classes for torch.load
        torch.serialization.add_safe_globals([
            DetectionModel, IterableSimpleNamespace,
            Sequential, ModuleList, ModuleDict,
            Conv2d, BatchNorm2d, SiLU, ReLU,
            MaxPool2d, Upsample,
            Conv, Concat,
            C1, C2, C3, C2f, Bottleneck, SPPF, DFL,
            Detect
        ])

        self.model = YOLO(weight_path)
        if device:
            self.model.to(device)
        self.model.fuse()
        self.conf = conf
        # class indices to detect; None → all classes
        self.classes = classes

    def __call__(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=0.5,
            classes=self.classes,
            verbose=False
        )[0]
        return results.boxes.xyxy.cpu().numpy()
