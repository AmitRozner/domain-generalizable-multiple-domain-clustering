from spice.model.feature_modules.resnet_all import resnet34, resnet18

class build_ResNet34:
    def __init__(self, **kwargs):
        pass

    def build(self, num_classes):
        return resnet34(num_classes=num_classes)

class build_ResNet18:
    def __init__(self, **kwargs):
        pass

    def build(self, num_classes):
        return resnet18(num_classes=num_classes)