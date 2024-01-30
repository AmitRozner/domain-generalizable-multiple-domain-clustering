model_name = "spice_self"
pre_model = "./results/domainnet_small/moco/moco_dl0.01_lr0.01_adamw_epoch999_loss0.67.pth"
embedding = "./results/domainnet_small/embedding/feas_moco_512_l2.npy"
resume = "".format(model_name)
model_type = "resnet18"
num_head = 10
num_workers = 16
device_id = 4
num_train = 5
num_cluster = 20
batch_size = 128
target_sub_batch_size = 64
train_sub_batch_size = 64
batch_size_test = 100
num_trans_aug = 1
num_repeat = 8
fea_dim = 512

att_conv_dim = num_cluster
att_size = 7
center_ratio = 0.5
sim_center_ratio = 0.9
epochs = 100
world_size = 1
workers = 4
rank = 0
dist_url = 'tcp://localhost:10001'
dist_backend = "nccl"
seed = None
gpu = None
multiprocessing_distributed = True
use_domain_classifier = True
start_epoch = 0
print_freq = 1
test_freq = 1
eval_ent = False
eval_ent_weight = 0

data_train = dict(
    type="domainnet_small_emb",
    root_folder="./datasets/domainnet_small",
    embedding=embedding,
    split="train+test",
    ims_per_batch=batch_size,
    shuffle=True,
    resize=(256, 256),
    aspect_ratio_grouping=False,
    train=True,
    show=False,
    trans1=dict(
        aug_type="weak",
        crop_size=256,
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        resize=300,
    ),

    trans2=dict(
        aug_type="scan",
        crop_size=256,
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        num_strong_augs=4,
        cutout_kwargs=dict(n_holes=1,
                           length=32,
                           random=True),
        resize=300,
    ),
    domain_names='clipart_sketch',
    use_moco_v3=False,
)

data_test = dict(
    type="domainnet_small_emb",
    root_folder="./datasets/domainnet_small",
    embedding=embedding,
    split="train+test",
    shuffle=False,
    ims_per_batch=50,
    aspect_ratio_grouping=False,
    train=False,
    resize=(256, 256),
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
    domain_names='clipart_sketch',
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
                                loss_weight=dict(loss_cls=1),
                                iter_start=epochs,
                                iter_up=epochs,
                                iter_down=epochs,
                                iter_end=epochs,
                                ratio_start=1.0,
                                ratio_end=1.0,
                                center_ratio=center_ratio,
                                )]*num_head,
              ),
    domain_head= dict(type="domain_head", feature_dim=fea_dim, loss_weight=0.5),
    model_type="moco",
    pretrained=pre_model,
    freeze_conv=True,
)


solver = dict(
    type="adam",
    base_lr=0.005,
    bias_lr_factor=1,
    weight_decay=0,
    weight_decay_bias=0,
    target_sub_batch_size=target_sub_batch_size,
    batch_size=batch_size,
    train_sub_batch_size=train_sub_batch_size,
    num_repeat=num_repeat,
)

results = dict(
    output_dir="./results/domainnet_small/{}".format(model_name),
)
