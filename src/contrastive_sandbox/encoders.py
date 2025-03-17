from torchsig.models import XCiTClassifier
import torch.nn as nn
from torch import no_grad

class XCiT(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        self.backbone = XCiTClassifier(input_channels=input_channels, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)
    
    def predict(self, x):
        with no_grad():
            return self.forward(x)
