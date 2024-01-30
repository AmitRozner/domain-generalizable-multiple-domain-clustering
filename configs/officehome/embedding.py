model_name = "embedding"
weight = "./results/officehome/moco/moco_dl0.01_lr0.01_adamw_epoch999_loss0.67.pth"
model_type = "resnet18"
device = 0
num_cluster = 65
batch_size = 1024
world_size = 1
workers = 4
rank = 0
dist_url = 'tcp://localhost:10001'
dist_backend = "nccl"
seed = None
gpu = None
multiprocessing_distributed = True
use_domain_classifier = True

data_test = dict(
    type="officehome",
    root_folder="./datasets/officehome",
    split="train+test",
    shuffle=False,
    resize=(256, 256),
    ims_per_batch=50,
    aspect_ratio_grouping=False,
    train=False,
    show=False,
    trans1=dict(
        aug_type="test_resize",
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        size=256,
    ),
    trans2=dict(
        aug_type="test_resize",
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        size=256,
    ),
    domain_names='amazon_webcam',
    use_moco_v3=False,
)


model_sim = dict(
    type=model_type,
    num_classes=128,
    in_channels=3,
    in_size=256,
    batchnorm_track=True,
    test=False,
    feature_only=True,
    pretrained=weight,
    model_type="moco_sim_feature",
)


results = dict(
    output_dir="./results/officehome/{}".format(model_name),
)