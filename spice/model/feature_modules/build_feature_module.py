from .convnet import ConvNet
from .mlp import MLP
from .cluster_resnet import ClusterResNet
from .imagenet import ImageNet, ResNet34, ResNet18


def build_feature_module(fea_cfg_ori):
    fea_cfg = fea_cfg_ori.copy()
    fea_type = fea_cfg.pop("type")
    if fea_type == "mlp":
        return MLP(**fea_cfg)
    elif fea_type == "convnet":
        return ConvNet(**fea_cfg)
    elif fea_type == "clusterresnet":
        return ClusterResNet(**fea_cfg)
    elif fea_type == "resnet18":
        return ResNet18(**fea_cfg)
    elif fea_type == "imagenet":
        return ImageNet(**fea_cfg)
    elif fea_type == 'resnet34':
        return ResNet34(**fea_cfg)
    else:
        raise TypeError