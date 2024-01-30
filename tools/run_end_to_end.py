import os
import argparse
import sys

sys.path.insert(0, './')
from spice.config import Config
from tools.train_moco import main as moco_main
from tools.pre_compute_embedding import main as pc_embedding_main
from tools.train_self_v2 import main as train_self_v2_main
import torch
import random
import numpy as np
import wandb

parser = argparse.ArgumentParser(description='Domain DomainGMDC')
parser.add_argument('--data_type', default='office31', help='path to dataset')
parser.add_argument('--num_cluster', default=31, type=int, help="Number of classes in the data")
parser.add_argument('--data', metavar='DIR', default='./datasets/',
                    help='path to dataset')
parser.add_argument('--feature_dim', default=512, type=int,
                    help='Dimension of features')
parser.add_argument('--domain_loss_weight', default=0.1, type=float,
                    help='Weight for domain part of loss function')
parser.add_argument('--all', default=1, type=int,
                    help='1 denotes using both train and test data')
parser.add_argument('--img_size', default=224, type=int,
                    help='image size')
parser.add_argument('--use_domain_classifier', action='store_false',
                    help='whether to also train parameters in backbone')
parser.add_argument('--root_save_folder', metavar='DIR', default='./results/',
                    help='path to dataset')
parser.add_argument('--save-freq', default=500, type=int, metavar='N',
                    help='frequency of saving model')
parser.add_argument('--arch', metavar='ARCH', default='resnet34')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--train_self_batch_size', default=128, type=int,
                    metavar='N')
parser.add_argument('--embedding_batch_size', default=1024, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')  # 0.015
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--load_weights', default='', type=str, metavar='PATH',
                    help='path to load weights')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_false',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--domain_warmup_num_epochs', default=0, type=int,
                    help='Freeze backbone for several epochs to train domain head')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_false',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_false',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_false',
                    help='use cosine lr schedule')

## additions
parser.add_argument('--optimizer', default="sgd", type=str)
parser.add_argument('--use_moco_v3', action='store_true')
parser.add_argument('--use_moco_v3_aug', action='store_true', help="use only the aug of mocov3")
parser.add_argument('--domain_names', default='amazon_webcam_dslr',
                    help='domain names sep by _')
parser.add_argument('--val_domains', default='', help='validation domain names sep by _')
parser.add_argument('--train_self_lr', default=0.001, type=float)
parser.add_argument('--wandb_run_name', default='',
                    help='_domain_name_domain_loss_seed')
parser.add_argument('--multi_q', action='store_true', help='if true would have a Q for every dataset (domain)')
parser.add_argument('--use_wandb', action='store_true')
parser.add_argument('--dl_lastlayer', action='store_true',
                    help='if true domain loss on last layer an not on the embedding')
parser.add_argument('--equalize_domain_sample_select', action='store_true', help='Equalize selected samples by domain')
parser.add_argument('--balance_moco_domains', action='store_true', help='Balance loaded data by domain in moco stage')
parser.add_argument('--balance_train_self_domains', action='store_true', help='Balance loaded data by domain in train '
                                                                              'self v2 stage')
parser.add_argument('--soft_balance', action='store_true',
                    help='Random uniform choice of domain for getitem. When false and balance is enabled it will return '
                         'an example from each of the domains resulting in more iterations')
parser.add_argument('--keep_strong_heads', action='store_true')
parser.add_argument('--epoch2remove_weak_heads', default=30, type=int)
parser.add_argument('--remove_weak_heads_every_n_epochs', default=10, type=int)
parser.add_argument('--use_domain_in_consistency', action='store_true', help="use domain label when calculating "
                                                                             "reliable labels")
parser.add_argument('--domain_size_layers', nargs='+', type=int, default=[100],
                    help="512 returns to last worked domain head, try: 2048,1024,512,256")
parser.add_argument('--center_based_truncate', action='store_true',
                    help="Choose reliable labels based on cluster center")
parser.add_argument('--center_based_truncate_cos', action='store_true',
                    help="Choose reliable labels based on cluster center")
parser.add_argument('--style_transfer', action='store_true')
parser.add_argument('--p_style_transfer', default=0.3, type=float, help="Amount of style transfer images [0,1]")
parser.add_argument('--self_smoothing', default=0.0, type=float, help="float should be around 0.9-0.995")
parser.add_argument('--select_samples_max_k', type=int, default=10)
parser.add_argument('--self_num_repeat', type=int, default=8)
parser.add_argument('--self_epochs', type=int, default=100)
parser.add_argument('--local_consistency_inbase_domain', action='store_true')
parser.add_argument('--use_edges', action='store_true')
parser.add_argument('--local_cons_num_neighbor', type=int, default=10)
parser.add_argument('--local_cons_ratio_confident', default=0.99, type=float)
parser.add_argument('--self_unfreeze_bb_at_epoch', default=100000000, type=int)
parser.add_argument('--select_samples_v2', action='store_true')
parser.add_argument('--domains2comparewith', default="", type=str)
parser.add_argument('--heads2keep', type=int, default=10)
parser.add_argument('--self_grad_accum', type=int, default=1)
parser.add_argument('--p_bcd_augment', default=0.0, type=float,
                    help="probability to augment to the basic common domain out of style transfer in training")
parser.add_argument('--moco_p_bcd_augment', default=0.0, type=float,
                    help="probability to augment to the basic common domain out of style transfer in pre training")
parser.add_argument('--p_suprise_bcd', default=0.0, type=float)
parser.add_argument('--pred_based_smoothing', action='store_true')

def main():
    args = parser.parse_args()
    run_deterministic(args)
    if args.use_edges:
        args.style_transfer = True

    ##### 1. Pre-training DomainGMDC
    resize = int(args.img_size * 1.15)
    args.wandb_run_name = f'{args.wandb_run_name}_{args.domain_names}_val_{args.val_domains}_{args.domain_loss_weight}_seed_{args.seed}_mq_' \
                          f'{args.multi_q}_balance_m_tsv2_{args.balance_moco_domains}_{args.balance_train_self_domains}' \
                          f'_eql_sams_{args.equalize_domain_sample_select}_tsv2_lr_{args.train_self_lr}_soft_balance_{args.soft_balance}' \
                          f'_center_based_truncate_{args.center_based_truncate}'
    args.root_save_folder = os.path.join(args.root_save_folder, args.data_type, args.wandb_run_name)
    args.save_folder = os.path.join(args.root_save_folder, 'moco')
    args.type = args.data_type
    args.root_folder = os.path.join(args.data, args.data_type)
    moco_main(args)

    # ##### 2. Precompute embedding features
    cfg_embedding = Config.fromfile(os.path.join("./configs/", args.data_type, 'embedding.py'))
    cfg_embedding.weight = os.path.join(args.save_folder, "checkpoint_select.pth.tar")
    cfg_embedding.model_type = args.arch
    cfg_embedding.model_sim.type = args.arch
    cfg_embedding.model_sim.pretrained = cfg_embedding.weight
    cfg_embedding.batch_size = args.embedding_batch_size
    cfg_embedding.workers = args.workers
    cfg_embedding.data_test.domain_names = args.domain_names
    cfg_embedding.results.output_dir = os.path.join(args.root_save_folder, "embedding")
    # cfg_embedding.data_test.root_folder = args.data
    cfg_embedding.data_test.type = args.data_type
    cfg_embedding.data_test.num_cluster = args.num_cluster
    cfg_embedding.data_test.root_folder = os.path.join(args.data, args.data_type)
    cfg_embedding.gpu = 0
    cfg_embedding.dist_url = args.dist_url
    cfg_embedding.data_test.resize = (resize, resize)
    cfg_embedding.data_test.trans1.size = args.img_size
    cfg_embedding.data_test.trans2.size = args.img_size
    cfg_embedding.data_test.domains2comparewith = args.domains2comparewith
    pc_embedding_main(cfg_embedding)

    ##### 3. Train DomainGMDC
    cfg_self = Config.fromfile(os.path.join("./configs/", args.data_type, 'spice_self.py'))
    cfg_self.pre_model = cfg_embedding.weight
    cfg_self.batch_size = args.train_self_batch_size
    cfg_self.solver.batch_size = cfg_self.batch_size
    cfg_self.target_sub_batch_size = cfg_self.solver.batch_size // 8
    cfg_self.train_sub_batch_size = cfg_self.solver.batch_size // 8
    cfg_self.solver.target_sub_batch_size = cfg_self.target_sub_batch_size
    cfg_self.solver.train_sub_batch_size = cfg_self.train_sub_batch_size
    cfg_self.batch_size_test = 16
    cfg_self.embedding = os.path.join(args.root_save_folder, "embedding", "feas_moco_512_l2.npy")
    cfg_self.model_type = args.arch
    cfg_self.num_cluster = args.num_cluster
    cfg_self.att_conv_dim = args.num_cluster
    cfg_self.model.feature.num_classes = args.num_cluster

    for head in cfg_self.model.head.multi_heads:
        head.classifier.num_neurons[-1] = args.num_cluster
        head.num_cluster = args.num_cluster

    cfg_self.data_test.domain_names = args.domain_names
    cfg_self.data_train.domain_names = args.domain_names
    cfg_self.data_train.embedding = cfg_self.embedding
    cfg_self.data_test.embedding = cfg_self.embedding
    cfg_self.data_train.ims_per_batch = cfg_self.batch_size
    cfg_self.data_test.root_folder = os.path.join(args.data, args.data_type)
    cfg_self.data_train.root_folder = os.path.join(args.data, args.data_type)
    cfg_self.data_train.num_cluster = args.num_cluster
    cfg_self.data_test.num_cluster = args.num_cluster
    cfg_self.data_test.type = args.data_type + "_emb"
    cfg_self.data_train.type = args.data_type + "_emb"
    cfg_self.gpu = args.gpu
    cfg_self.use_domain_classifier = args.use_domain_classifier
    cfg_self.results.output_dir = os.path.join(args.root_save_folder, "spice_self")
    cfg_self.model.feature.type = cfg_self.model_type
    cfg_self.model.pretrained = cfg_embedding.weight
    cfg_self.all = 1
    cfg_self.epochs = args.self_epochs
    cfg_self.wandb_run_name = args.wandb_run_name
    cfg_self.dist_url = args.dist_url
    cfg_self.use_wandb = args.use_wandb
    cfg_self.equalize_domain_sample_select = args.equalize_domain_sample_select
    cfg_self.balance_train_self_domains = args.balance_train_self_domains
    cfg_self.keep_strong_heads = args.keep_strong_heads
    cfg_self.epoch2remove_weak_heads = args.epoch2remove_weak_heads
    cfg_self.remove_weak_heads_every_n_epochs = args.remove_weak_heads_every_n_epochs
    cfg_self.solver.base_lr = args.train_self_lr
    cfg_self.seed = args.seed
    cfg_self.model.domain_head.domain_size_layers = args.domain_size_layers
    cfg_self.soft_balance = args.soft_balance
    cfg_self.data_train.resize = (resize, resize)
    cfg_self.data_train.trans1.crop_size = args.img_size
    cfg_self.data_train.trans2.crop_size = args.img_size
    cfg_self.data_train.trans1.resize = resize
    cfg_self.data_train.trans2.resize = resize
    cfg_self.data_test.resize = (resize, resize)
    cfg_self.data_test.trans1.size = args.img_size
    cfg_self.data_test.trans2.size = args.img_size
    cfg_self.model.self_smoothing = args.self_smoothing
    cfg_self.style_transfer = args.style_transfer
    cfg_self.solver.num_repeat = args.self_num_repeat
    cfg_self.p_style_transfer = args.p_style_transfer
    cfg_self.use_edges = args.use_edges
    cfg_self.select_samples_max_k = args.select_samples_max_k
    cfg_self.unfreeze_bb_at_epoch = args.self_unfreeze_bb_at_epoch
    cfg_self.select_samples_v2 = args.select_samples_v2
    cfg_self.data_train.domains2comparewith = args.domains2comparewith
    cfg_self.data_test.domains2comparewith = args.domains2comparewith
    cfg_self.heads2keep = args.heads2keep
    cfg_self.self_grad_accum = args.self_grad_accum
    cfg_self.p_bcd_augment = args.p_bcd_augment
    cfg_self.p_suprise_bcd = args.p_suprise_bcd
    cfg_self.pred_based_smoothing = args.pred_based_smoothing
    train_self_v2_main(cfg_self)

def run_deterministic(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)


if __name__ == '__main__':
    main()
