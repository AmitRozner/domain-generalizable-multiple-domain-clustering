from .sem_head import SemHead
from .sem_head_multi import SemHeadMulti
from .domain_head import DomainClassifierReverse


def build_head(args, head_cfg_ori, gpu=None,**kwargs):
    head_cfg = head_cfg_ori.copy()
    if "self_smoothing" in kwargs:
        head_cfg["self_smoothing"]=kwargs["self_smoothing"]
    head_type = head_cfg.pop("type")
    if head_type == "sem":
        return SemHead(**head_cfg)
    elif head_type == "sem_multi":
        return SemHeadMulti(args, **head_cfg)
    elif head_type == "domain_head":
        return DomainClassifierReverse(head_cfg["domain_size_layers"],head_cfg['feature_dim'], head_cfg['loss_weight'], gpu)
    else:
        raise TypeError
