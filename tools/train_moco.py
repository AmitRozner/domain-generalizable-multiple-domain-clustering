#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys

sys.path.insert(0, './')
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from spice.model.feature_modules.cluster_resnet import ClusterResNet
from spice.model.feature_modules.resnet_all import resnet34, resnet18
import numpy as np
import moco.loader
import moco.builder
from moco.stl10 import STL10
from moco.cifar import CIFAR10, CIFAR100
from spice.data.build_dataset import build_dataset
from spice.utils.miscellaneous import choose_num_workers
import copy
import traceback
from adain.adain import AdaIN
import glob
from tools.train_self_v2 import internal_read_image

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_type', default='office31', help='path to dataset')
parser.add_argument('--data', metavar='DIR', default='./datasets/office31',
                    help='path to dataset')
parser.add_argument('--feature_dim', default=512, type=int,
                    help='Dimension of features')
parser.add_argument('--domain_loss_weight', default=0.1, type=float,
                    help='Weight for domain part of loss function')
parser.add_argument('--all', default=1, type=int,
                    help='1 denotes using both train and test data')
parser.add_argument('--img_size', default=256, type=int,
                    help='image size')
parser.add_argument('--use_domain_classifier', action='store_false',
                    help='whether to also train parameters in backbone')
parser.add_argument('--save_folder', metavar='DIR', default='./results/office31/moco',
                    help='path to dataset')
parser.add_argument('--save-freq', default=50, type=int, metavar='N',
                    help='frequency of saving model')
parser.add_argument('--arch', metavar='ARCH', default='resnet34')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
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
                    help='path to latest checkpoint (default: none)')  # ./results/stl10/moco/checkpoint_last.pth.tar
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
parser.add_argument('--seed', default=None, type=int,
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
parser.add_argument('--domain_names', default='amazon_webcam',
                    help='domain names sep by _')
parser.add_argument('--use_wandb', action='store_true')
parser.add_argument('--wandb_run_name', default='',
                    help='_domain_name_domain_loss_seed')
parser.add_argument('--multi_q', action='store_true', help='if true would have a Q for every dataset (domain)')
parser.add_argument('--dl_lastlayer', action='store_true',
                    help='if true domain loss on last layer and not on the embedding')
parser.add_argument('--domain_size_layers', nargs='+', type=int, default=[512],
                    help="512 returns to last worked domain head, try: 2048,1024,512,256")
parser.add_argument('--style_transfer', action='store_true')
parser.add_argument('--p_style_transfer', default=0.3, type=float, help="Amount of style transfer images [0,1]")


def main(args=None):
    if args is None:
        args = parser.parse_args()
        args.type = args.data_type
        args.root_folder = os.path.join(args.data, args.data_type)

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    run_name = "moco_" + args.wandb_run_name

    if args.multi_q:
        assert args.balance_moco_domains, "Cannot use multi queue without balance domains"

    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.use_wandb and args.rank == 0:
        import wandb
        wandb.init(project="domain_dgmc", name=run_name, entity="user", config=vars(cfg))
    else:
        wandb = None

    try:
        main_func(args, ngpus_per_node, wandb)
    except Exception:
        print(traceback.print_exc(), file=sys.stderr)
    finally:
        if args.use_wandb and args.rank == 0:
            wandb.finish()


def main_func(args, ngpus_per_node, wandb):
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == "clusterresnet":
        base_model = ClusterResNet
    elif args.arch == "resnet18":
        base_model = resnet18
    elif args.arch == "resnet34":
        base_model = resnet34
    elif args.arch == "resnet18_cifar":
        base_model = resnet18_cifar
    else:
        base_model = models.__dict__[args.arch]
    model = moco.builder.MoCo(
        base_model, args, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, input_size=args.img_size,
        num_domains=len(args.domain_names.split("_")))
    if len(args.val_domains) > 0:
        val_args = copy.deepcopy(args)
        val_args.domain_names = args.val_domains
        val_model = moco.builder.MoCo(base_model, val_args, val_args.moco_dim, val_args.moco_k, val_args.moco_m,
                                      val_args.moco_t, val_args.mlp, input_size=val_args.img_size,
                                      num_domains=len(val_args.domain_names.split("_")))
    else:
        val_model = None
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            if len(args.val_domains) > 0:
                val_model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
            if len(args.val_domains) > 0:
                val_model = torch.nn.parallel.DistributedDataParallel(val_model, device_ids=[args.gpu],
                                                                      find_unused_parameters=False)
        else:
            model.cuda()
            if len(args.val_domains) > 0:
                val_model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
            if len(args.val_domains) > 0:
                val_model = torch.nn.parallel.DistributedDataParallel(val_model, find_unused_parameters=False)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        if len(args.val_domains) > 0:
            val_model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    if len(args.val_domains) > 0:
        val_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        val_criterion = None
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                      weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    elif args.load_weights:
        if os.path.isfile(args.load_weights):
            print("=> loading checkpoint '{}'".format(args.load_weights))
            if args.gpu is None:
                checkpoint = torch.load(args.load_weights)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.load_weights, map_location=loc)
            from collections import OrderedDict
            new_dict = OrderedDict()
            layers_not_to_load = ["num_batches_tracked", "module.queue", "module.queue_idx", "domain_head"]
            for k, v in checkpoint["state_dict"].items():
                k_in_unwanted_layers = []
                for layer in layers_not_to_load:
                    k_in_unwanted_layers.append(layer in k)
                if any(k_in_unwanted_layers):
                    continue
                if v.shape[0] == 2 and "domain_classifier" not in k:
                    new_dict[k] = v[0]
                else:
                    new_dict[k] = v

            model.load_state_dict(new_dict, strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.load_weights, args.start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.load_weights))

    if args.style_transfer:
        adain_model = AdaIN(
            "adain_weights/decoder.pth",
            "adain_weights/vgg_normalised.pth",
            args.gpu,
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
        )
        if args.use_edges:
            edge_model = ToEdges()
        else:
            edge_model = None
    else:
        adain_model = None
        edge_model = None

    # Data loading code
    if args.data_type.startswith('mnist'):
        normalize = transforms.Normalize(mean=[0.464, 0.468, 0.42],
                                         std=[0.253, 0.238, 0.262])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    if args.use_moco_v3 or args.use_moco_v3_aug:
        augmentation1 = [
            transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        augmentation2 = [
            transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([moco.loader.Solarize()], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    if args.data_type == 'stl10':
        if args.all:
            split = "train+test+unlabeled"
        else:
            split = "train+unlabeled"
        train_dataset = STL10(os.path.join(args.data, args.data_type), split=split,
                              transform=moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    elif args.data_type == 'cifar10':
        train_dataset = CIFAR10(os.path.join(args.data, args.data_type), all=args.all,
                                transform=moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    elif args.data_type == 'cifar100':
        train_dataset = CIFAR100(os.path.join(args.data, args.data_type), all=args.all,
                                 transform=moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    elif args.data_type.startswith('office') or args.data_type.startswith('mnist') \
            or args.data_type.startswith('domainnet') or args.data_type.startswith('pacs'):
        if args.use_moco_v3:
            train_dataset = build_dataset(args, train_trans1=augmentation1, train_trans2=augmentation2,
                                          balance_datasets=args.balance_moco_domains)
        else:
            train_dataset = build_dataset(args,
                                          train_trans1=moco.loader.TwoCropsTransform(transforms.Compose(augmentation)),
                                          train_trans2=transforms.Compose([transforms.Resize((args.img_size,
                                                                                              args.img_size)),
                                                                           transforms.ToTensor()]),
                                          balance_datasets=args.balance_moco_domains)
        if len(args.val_domains) > 0:
            val_dataset = build_dataset(val_args,
                                        train_trans1=moco.loader.TwoCropsTransform(transforms.Compose(augmentation)),
                                        train_trans2=transforms.Compose([transforms.Resize((val_args.img_size,
                                                                                            val_args.img_size)),
                                                                         transforms.ToTensor()]),
                                        balance_datasets=val_args.balance_moco_domains)
        else:
            val_dataset = None
    else:
        raise TypeError

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    chosen_num_worker = choose_num_workers(generator, train_dataset, train_sampler, args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=chosen_num_worker, pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True, generator=generator, persistent_workers=True)
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=chosen_num_worker, pin_memory=True, drop_last=True,
                                                 generator=generator, persistent_workers=True)
    else:
        val_loader = None
    best_top1 = 0

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        if epoch > args.epochs - 15:
            break
        if epoch < args.domain_warmup_num_epochs:
            model.eval()

            if args.distributed:
                model.module.domain_head.train()
            else:
                model.domain_head.train()

            for name, param in model.named_parameters():
                if "domain_head" not in name:
                    param.requires_grad = False

        elif epoch == args.domain_warmup_num_epochs:
            print("===== Unfreezing backbone =====")
            for name, param in model.named_parameters():
                param.requires_grad = True

        if args.distributed:
            train_sampler.set_epoch(epoch)
        if not args.use_moco_v3:
            adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        top1 = train(train_loader, val_loader, model, val_model, criterion, val_criterion, optimizer, epoch, args,
                     wandb, adain_model, edge_model)
        is_best = False
        if best_top1 < top1:
            is_best = True
            best_top1 = top1

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            if is_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename='{}/checkpoint_select.pth.tar'.format(args.save_folder, epoch))

            if (epoch + 1) % args.save_freq == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.save_folder, epoch))

                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename='{}/checkpoint_last.pth.tar'.format(args.save_folder))

            if (epoch + 1) == args.epochs:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename='{}/checkpoint_final.pth.tar'.format(args.save_folder))
    if args.use_wandb and args.rank == 0:
        wandb.finish()


def train(train_loader, val_loader, model, eval_model, criterion, val_criterion, optimizer, epoch, args, wandb=None,
          adain_model=None, edge_model=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    val_top1 = AverageMeter('Acc@1', ':6.2f')
    val_losses = AverageMeter('Loss', ':.4e')
    domain_losses = AverageMeter('Domain_loss', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, domain_losses],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    iters_per_epoch = len(train_loader)

    for step, (images_first_aug, images_aug_v3, _, domain_labels, sample_idx) in enumerate(train_loader):
        domain_labels = torch.cat(domain_labels)
        sample_idx = torch.reshape(sample_idx.T, (len(domain_labels),))
        # measure data loading time
        data_time.update(time.time() - end)
        import matplotlib.pyplot as plt
        # plt.imsave("image1.jpg",transforms.ToPILImage()(images[0][0,:,:,:]))#, interpolation="bicubic")
        # plt.imsave("image2.jpg",transforms.ToPILImage()(images[1][0, :, :, :]))#, interpolation="bicubic")
        # x = images[0][0, :, :, :].permute(1, 2, 0).cpu().numpy()
        # img1  = np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
        # x = images[1][0, :, :, :].permute(1, 2, 0).cpu().numpy()
        # img2 = np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
        # plt.imsave("image1.jpg",img1)
        # plt.imsave("image2.jpg", img2)
        # stop
        if args.use_moco_v3:
            lr = adjust_learning_rate(optimizer, epoch + step / iters_per_epoch, args)
            moco_m = adjust_moco_momentum(epoch + step / iters_per_epoch, args)

        ####################
        # Generate style transferred images
        ####################
        if args.style_transfer:
            num_domains = len(args.domain_names.split("_"))
            all_domains = list(range(num_domains))
            style_transfer_images = []
            for domain_idx in all_domains:
                other_domains = all_domains.copy()
                other_domains.remove(domain_idx)
                traget_domain_style = np.random.choice(np.array(other_domains), 1)[0]
                # Transfer
                if edge_model is not None:
                    if random.random() < args.p_style_transfer:
                        xu_k_sty = adain_model(images_first_aug[domain_idx][0].cuda(args.gpu, non_blocking=True),
                                               images_first_aug[traget_domain_style][0].cuda(args.gpu,
                                                                                             non_blocking=True))
                    else:
                        xu_k_sty = edge_model(images_first_aug[domain_idx][0].cuda(args.gpu, non_blocking=True))

                else:
                    if random.random() < args.moco_p_bcd_augment:
                        sketch_style_images = glob.glob("./style_sketch_images/*")
                        img_idx = np.random.randint(0, len(sketch_style_images))
                        image_arr = torchvision.transforms.Resize((224, 224))(
                            internal_read_image(sketch_style_images[img_idx])).cuda()
                        image_arr[0] = ((image_arr[0] - torch.min(image_arr[0])) / (
                                torch.max(image_arr[0]) - torch.min(image_arr[0]) + 10e-10)).cuda() * 2
                        if image_arr.shape[0] == 1:
                            image_arr = image_arr.repeat(3, 1, 1)
                        xu_k_sty = adain_model(images_first_aug[domain_idx][0].cuda(args.gpu, non_blocking=True),
                                               image_arr.unsqueeze(dim=0).repeat(args.batch_size,
                                                                                 1, 1, 1))
                    else:
                        xu_k_sty = adain_model(images_first_aug[domain_idx][0].cuda(args.gpu, non_blocking=True),
                                               images_first_aug[traget_domain_style][0].cuda(args.gpu,
                                                                                             non_blocking=True))

                style_transfer_images.append(xu_k_sty)

        if args.gpu is not None:
            images = []
            num_domains = len(args.domain_names.split("_"))
            if args.use_moco_v3:
                images.append(torch.cat([images_first_aug[domain_idx] for domain_idx in
                                         range(num_domains)]).cuda(args.gpu, non_blocking=True))
                images.append(torch.cat([images_aug_v3[domain_idx] for domain_idx in
                                         range(num_domains)]).cuda(args.gpu, non_blocking=True))
            else:
                if args.balance_moco_domains:
                    images.append(torch.cat([images_first_aug[domain_idx][0] for domain_idx in
                                             range(num_domains)]).cuda(args.gpu, non_blocking=True))
                    images.append(torch.cat([images_first_aug[domain_idx][1] for domain_idx in
                                             range(num_domains)]).cuda(args.gpu, non_blocking=True))
                    if args.style_transfer:
                        sty_images = torch.cat([style_transfer_images[domain_idx] for domain_idx in
                                                range(num_domains)]).cuda(args.gpu, non_blocking=True)
                else:
                    images.append(images_first_aug[0][0].cuda(args.gpu, non_blocking=True))
                    images.append(images_first_aug[0][1].cuda(args.gpu, non_blocking=True))

            if args.style_transfer:
                chosen_style_ind = torch.tensor(
                    np.random.choice([0, 1], p=[1 - args.p_style_transfer, args.p_style_transfer],
                                     size=len(sty_images))).view(images[0].shape[0], 1, 1, 1).cuda(args.gpu,
                                                                                                   non_blocking=True)
                # style transfer will replace the strong augmentation
                images[1] = images[1] * (1 - chosen_style_ind) + sty_images * chosen_style_ind

            if args.use_domain_classifier:
                domain_labels = domain_labels.cuda(args.gpu, non_blocking=True)
                p = float(step + epoch * len(train_loader)) / args.epochs / len(train_loader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
            else:
                domain_labels = None
                alpha = 0

        # compute output
        output, target, domain_loss = model(im_q=images[0], im_k=images[1], domain_labels=domain_labels, alpha=alpha,
                                            q_selector=domain_labels, sample_idx=sample_idx)

        domain_losses.update(domain_loss.item(), images[0].size(0))
        loss = criterion(output, target)

        if args.use_domain_classifier:
            loss += domain_loss

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0].item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            progress.display(step)

    if val_loader is not None:
        for param_q, param_k in zip(model.named_parameters(), eval_model.named_parameters()):
            if "domain" not in param_q[0]:
                param_k[1].data.copy_(param_q[1].detach().data)  # initialize
                param_k[1].requires_grad = False  # not update by gradient for eval_net

        eval_model.eval()

        for step, (images_first_aug, images_aug_v3, _, domain_labels, sample_idx) in enumerate(val_loader):
            domain_labels = torch.cat(domain_labels)
            sample_idx = torch.reshape(sample_idx.T, (len(domain_labels),))
            if args.gpu is not None:
                images = []
                if args.use_moco_v3:
                    images.append(torch.cat([images_first_aug[domain_idx] for domain_idx in
                                             range(len(images_first_aug))]).cuda(args.gpu, non_blocking=True))
                    images.append(torch.cat([images_aug_v3[domain_idx] for domain_idx in
                                             range(len(images_first_aug))]).cuda(args.gpu, non_blocking=True))
                else:
                    images.append(torch.cat([images_first_aug[domain_idx][0] for domain_idx in
                                             range(len(images_first_aug))]).cuda(args.gpu, non_blocking=True))
                    images.append(torch.cat([images_first_aug[domain_idx][1] for domain_idx in
                                             range(len(images_first_aug))]).cuda(args.gpu, non_blocking=True))

                if args.use_domain_classifier:
                    domain_labels = domain_labels.cuda(args.gpu, non_blocking=True)
                    p = float(step + epoch * len(train_loader)) / args.epochs / len(train_loader)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                else:
                    domain_labels = None
                    alpha = 0

            # compute output
            with torch.no_grad():
                output, target, domain_loss = eval_model(im_q=images[0], im_k=images[1], domain_labels=domain_labels,
                                                         alpha=alpha, q_selector=domain_labels, sample_idx=sample_idx)

            val_acc1 = accuracy(output, target, topk=(1,))

            val_top1.update(val_acc1[0], images[0].size(0))
            val_loss = val_criterion(output, target)
            val_losses.update(val_loss.item(), images[0].size(0))

    if args.use_wandb and args.rank == 0:
        wandb.log({"acc1": top1.avg, "loss": losses.avg, "domain_losses": domain_losses.avg,
                   "lr": optimizer.param_groups[0]['lr'], "alpha": alpha})
        if val_loader is not None:
            wandb.log({"val_acc1": val_top1.avg, "val_loss": val_losses.avg})
    return top1.avg


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    if args.use_moco_v3:
        if epoch < args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs
        else:
            lr = args.lr * 0.5 * (
                    1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    else:
        lr = args.lr
        if args.cos:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        else:  # stepwise lr schedule
            for milestone in args.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(k * correct.shape[1]).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
