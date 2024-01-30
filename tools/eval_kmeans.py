import os
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from spice.utils.load_model_weights import load_model_weights
from spice.data.build_dataset import build_dataset
from spice.config import Config
from spice.model.build_model_sim import build_model_sim
from spice.utils.evaluation import calculate_acc
from fixmatch.models.fixmatch.rfixmatch_v1 import FixMatch

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_type', default='pacs')
parser.add_argument('--domain_names', default='cartoon')
parser.add_argument('--dataset_dir',
                    default='')
parser.add_argument('--num_cluster', default=7, type=int, help="Number of classes in the data")
parser.add_argument('--data_dir', default='./datasets')
parser.add_argument('--root_dir', default='results')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--arch', metavar='ARCH', default='resnet18')
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--img_size', default=224, type=int, help='image size')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--domains2comparewith', default='')


def compute_kmeans_acc(args):
    cfg_embedding = Config.fromfile(os.path.join(f"{os.getcwd()}/configs/", args.data_type, 'embedding.py'))
    root_data_folder = os.path.join(args.root_dir, args.data_type, args.dataset_dir)
    cfg_embedding.weight = os.path.join(root_data_folder, "moco", "checkpoint_select.pth.tar")
    cfg_embedding.model_type = args.arch
    cfg_embedding.model_sim.type = args.arch
    cfg_embedding.model_sim.pretrained = os.path.join(root_data_folder, "moco", "checkpoint_select.pth.tar")
    cfg_embedding.batch_size = args.batch_size
    cfg_embedding.workers = 1
    cfg_embedding.data_test.domain_names = args.domain_names
    cfg_embedding.results.output_dir = os.path.join(root_data_folder, "embedding")
    cfg_embedding.data_test.type = args.data_type
    cfg_embedding.data_test.num_cluster = args.num_cluster
    cfg_embedding.data_test.root_folder = os.path.join(args.data_dir, args.data_type)
    cfg_embedding.gpu = 0
    cfg_embedding.dist_url = args.dist_url
    resize = int(args.img_size * 1.15)
    cfg_embedding.data_test.resize = (resize, resize)
    cfg_embedding.data_test.trans1.size = args.img_size
    cfg_embedding.data_test.trans2.size = args.img_size
    cfg_embedding.data_test.domains2comparewith = args.domains2comparewith

    # create model
    model_sim = build_model_sim(cfg_embedding.model_sim)
    torch.cuda.set_device(cfg_embedding.gpu)
    model_sim.cuda(cfg_embedding.gpu)
    load_model_weights(model_sim, cfg_embedding.model_sim.pretrained, "moco_for_kmeans")
    dataset_val = build_dataset(cfg_embedding.data_test)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=cfg_embedding.batch_size, shuffle=False,
                                             num_workers=4,
                                             persistent_workers=True)
    pool = nn.AdaptiveAvgPool2d(1)

    feas_sim = []
    targets = []
    for _, (images, _, y, _, _) in enumerate(val_loader):
        images = torch.cat(images).to(cfg_embedding.gpu, non_blocking=True)
        print(images.shape)
        targets.append(y)
        with torch.no_grad():
            feas_sim_i = model_sim(images)
            if len(feas_sim_i.shape) == 4:
                feas_sim_i = pool(feas_sim_i)
                feas_sim_i = torch.flatten(feas_sim_i, start_dim=1)
            feas_sim_i = nn.functional.normalize(feas_sim_i, dim=1)
            feas_sim.append(feas_sim_i.cpu())

    feas_sim = torch.cat(feas_sim, dim=0)
    feas_sim = feas_sim.numpy()
    kmeans_acc = []
    for i in range(20):
        k_means = KMeans(n_clusters=cfg_embedding.num_cluster).fit(feas_sim)
        k_means_pred = k_means.labels_
        targets_con = np.concatenate([target[0] for target in targets])
        acc, ari, nmi = FixMatch.calc_metrics(targets_con, k_means_pred)
        kmeans_acc.append(acc)
    print(kmeans_acc)
    print(f"K-Means acc:{sum(kmeans_acc) / len(kmeans_acc)}")
    return acc, ari, nmi


def main(dataset_dir="", data_type="", domain_names="", num_cluster="", args=None):
    if args is None:
        args = parser.parse_args()
    else:
        args.root_dir = args.root_save_folder.split('/')[1]
        args.data_dir = args.root_folder.split('/')[1]

    if dataset_dir != "":
        args.dataset_dir = dataset_dir
    if data_type != "":
        args.data_type = data_type
    if domain_names != "":
        args.domain_names = domain_names
    if num_cluster != "":
        args.num_cluster = num_cluster

    acc = compute_kmeans_acc(args)
    return acc


if __name__ == '__main__':
    acc = main()
