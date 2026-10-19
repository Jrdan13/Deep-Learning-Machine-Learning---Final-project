import segmentation_models_pytorch as smp

NUM_CLASSES = 11


def build_model(encoder="resnet34", pretrained=True):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=NUM_CLASSES,
    )