model_name = "eval"
weight = './results/officehome/spice_self/checkpoint_select.pth.tar'
model_type = "resnet18"
device_id = 0
num_cluster = 65
batch_size = 100
fea_dim = 512
center_ratio = 0.5
world_size = 1
workers = 4
rank = 0
dist_url = 'tcp://localhost:10001'
dist_backend = "nccl"
seed = None
gpu = None
multiprocessing_distributed = True

data_test = dict(
    type="officehome_emb",
    root_folder="./datasets/officehome",
    embedding=None,
    split="train+test",
    shuffle=False,
    resize=(256, 256),
    ims_per_batch=1,
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

model = dict(
    feature=dict(
        type=model_type,
        num_classes=num_cluster,
        in_channels=3,
        in_size=256,
        batchnorm_track=True,
        test=False,
        feature_only=True
    ),

    head=dict(type="sem_multi",
              multi_heads=[dict(classifier=dict(type="mlp", num_neurons=[fea_dim, fea_dim, num_cluster], last_activation="softmax"),
                                feature_conv=None,
                                num_cluster=num_cluster,
                                ratio_start=1,
                                ratio_end=1,
                                center_ratio=center_ratio,
                                )]*1,
              ratio_confident=0.90,
              num_neighbor=10,
              ),
    domain_head= dict(type="domain_head", feature_dim=fea_dim, loss_weight=0.5,domain_size_layers=[512]),#
    model_type="moco_select",
    pretrained=weight,
    head_id=0,
    freeze_conv=True,
)

results = dict(
    output_dir="./results/officehome/{}".format(model_name),
)