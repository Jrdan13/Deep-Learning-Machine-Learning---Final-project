import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder

NUM_CLASSES = 11
ARCHS = ["unet", "fcn", "deeplabv3plus"]  # add new models here


class FCN8s(nn.Module):
    """FCN-8s (Long et al., 2015) with a ResNet34 encoder instead of VGG16.
    Class scores from strides 32, 16 and 8 are upsampled and summed, then upsampled 8x
    (bilinear interpolation)."""

    def __init__(self, encoder="resnet34", pretrained=True, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = get_encoder(encoder, in_channels=3, depth=5,
                                   weights="imagenet" if pretrained else None)
        ch = self.encoder.out_channels
        self.score32 = nn.Conv2d(ch[5], num_classes, 1)
        self.score16 = nn.Conv2d(ch[4], num_classes, 1)
        self.score8 = nn.Conv2d(ch[3], num_classes, 1)

    @staticmethod
    def _up(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        f = self.encoder(x)
        s = self._up(self.score32(f[5]), f[4].shape[-2:]) + self.score16(f[4])
        s = self._up(s, f[3].shape[-2:]) + self.score8(f[3])
        return self._up(s, x.shape[-2:])


def build_model(arch="unet", encoder="resnet34", pretrained=True):
    if arch == "unet":
        return smp.Unet(encoder_name=encoder,
                        encoder_weights="imagenet" if pretrained else None,
                        in_channels=3, classes=NUM_CLASSES)
    if arch == "fcn":
        return FCN8s(encoder, pretrained)
    
    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(encoder_name=encoder,
                                 encoder_weights="imagenet" if pretrained else None,
                                 in_channels=3, classes=NUM_CLASSES)
    raise ValueError(f"Unknown arch '{arch}'. Known: {ARCHS}")