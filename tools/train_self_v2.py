import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys
import torchvision
import glob
sys.path.insert(0, './')
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from spice.config import Config
from spice.data.build_dataset import build_dataset
from spice.model.sim2sem import Sim2Sem
from spice.solver import make_lr_scheduler, make_optimizer
from spice.utils.miscellaneous import mkdir, save_config, choose_num_workers
from spice.utils.evaluation import calculate_acc, calculate_nmi, calculate_ari
from spice.utils.load_model_weights import load_model_weights
from spice.utils.logger import setup_logger
import logging
from spice.utils.comm import get_rank
import numpy as np
from tqdm import tqdm
import wandb
import pandas as pd
import traceback
from adain.adain import AdaIN
from skimage import feature
from skimage.morphology import dilation, disk
import PIL
import torchvision.transforms.functional as transform
def internal_read_image(image_path):
    PIL_image = PIL.Image.open(image_path)
    tensor_image = transform.to_tensor(PIL_image)
    return tensor_image
class ToEdges(torch.nn.Module):
    def __init__(self, sigma='1.0', dil='0'):
        super().__init__()
        self.sigma = [float(x) for x in sigma.split(',')]
        self.dil = [int(x) for x in dil.split(',')]

    def forward(self, imgs):
        img_out_list = []
        for img_orig in imgs:
            img = torch.mean(img_orig, dim=0).cpu().numpy()
            sig = self.sigma[torch.randint(len(self.sigma), (1,))[0]]
            edg = feature.canny(img, sigma=sig)
            dl = self.dil[torch.randint(len(self.dil), (1,))[0]]
            if dl > 0:
                edg = dilation(edg, disk(dl))
            img_out = torch.Tensor(edg).unsqueeze(dim=0).repeat([3, 1, 1])
            img_out_list.append(img_out)
        img_out_list = torch.cat([img.unsqueeze(0) for img in img_out_list], dim=0).cuda()
        return img_out_list

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
NO_ACCURACY = -0.01

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    "--config-file",
    default="./configs/office31/spice_self.py",
    metavar="FILE",
    help="path to config file",
    type=str,
)

parser.add_argument(
    "--all",
    default=1,
    type=int,
)


def main(cfg=None):
    if cfg is None:
        args = parser.parse_args()
        cfg = Config.fromfile(args.config_file)
        cfg.all = args.all

    output_dir = cfg.results.output_dir
    if output_dir:
        mkdir(output_dir)

    output_config_path = os.path.join(output_dir, 'config.py')
    save_config(cfg, output_config_path)


    if cfg.all:
        cfg.data_train.split = "train+test"
        cfg.data_train.all = True
        cfg.data_test.split = "train+test"
        cfg.data_test.all = True
    else:
        cfg.data_train.split = "train"
        cfg.data_train.all = False
        cfg.data_train.train = True
        cfg.data_test.split = "train"
        cfg.data_test.all = False
        cfg.data_test.train = True

    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if cfg.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if cfg.dist_url == "env://" and cfg.world_size == -1:
        cfg.world_size = int(os.environ["WORLD_SIZE"])

    cfg.distributed = cfg.world_size > 1 or cfg.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if cfg.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        cfg.world_size = ngpus_per_node * cfg.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg.copy()))
    else:
        # Simply call main_worker function
        main_worker(cfg.gpu, ngpus_per_node, cfg)


def copy_params(k, buffer):
    k.classifier.lin1.weight.data.copy_(buffer[0].detach().clone())
    k.classifier.lin1.bias.data.copy_(buffer[1].detach().clone())
    k.classifier.lin2.weight.data.copy_(buffer[2].detach().clone())
    k.classifier.lin2.bias.data.copy_(buffer[3].detach().clone())


def copy_params_evolution(k, buffers):
    len_b = len(buffers)
    for j in range(len(buffers)):
        if j == 0:
            lin1_w = buffers[j][0].detach().clone()
            lin1_b = buffers[j][1].detach().clone()
            lin2_w = buffers[j][2].detach().clone()
            lin2_b = buffers[j][3].detach().clone()
        else:
            lin1_w += buffers[j][0].detach().clone()
            lin1_b += buffers[j][1].detach().clone()
            lin2_w += buffers[j][2].detach().clone()
            lin2_b += buffers[j][3].detach().clone()
    lin1_w, lin1_b, lin2_w, lin2_b = buffers[j][0] / len_b, buffers[j][1] / len_b, buffers[j][2] / len_b, buffers[j][
        3] / len_b
    k.classifier.lin1.weight.data.copy_(lin1_w)
    k.classifier.lin1.bias.data.copy_(lin1_b)
    k.classifier.lin2.weight.data.copy_(lin2_w)
    k.classifier.lin2.bias.data.copy_(lin2_b)


def copy_params_evolution_random_weights(k, buffers):
    len_b = len(buffers)
    weights = np.random.random(len_b)
    weights = weights / sum(weights)
    for j in range(len(buffers)):
        if j == 0:
            lin1_w = weights[j] * buffers[j][0].detach().clone()
            lin1_b = weights[j] * buffers[j][1].detach().clone()
            lin2_w = weights[j] * buffers[j][2].detach().clone()
            lin2_b = weights[j] * buffers[j][3].detach().clone()
        else:
            lin1_w += weights[j] * buffers[j][0].detach().clone()
            lin1_b += weights[j] * buffers[j][1].detach().clone()
            lin2_w += weights[j] * buffers[j][2].detach().clone()
            lin2_b += weights[j] * buffers[j][3].detach().clone()

    k.classifier.lin1.weight.data.copy_(lin1_w)
    k.classifier.lin1.bias.data.copy_(lin1_b)
    k.classifier.lin2.weight.data.copy_(lin2_w)
    k.classifier.lin2.bias.data.copy_(lin2_b)

def keep_the_strong_heads_func(model, heads2keep):
    head2keep_w_buffer = []
    for i, k in enumerate(model.module.head.children()):
        if i in heads2keep:
            head2keep_w_buffer.append(
                [k.classifier.lin1.weight, k.classifier.lin1.bias, k.classifier.lin2.weight, k.classifier.lin2.bias])
    for i, k in enumerate(model.module.head.children()):
        if i not in heads2keep:
            copy_params_evolution_random_weights(k, head2keep_w_buffer)
    print("Copied params from:", heads2keep, "heads!")


def main_worker(gpu, ngpus_per_node, cfg):
    cfg.gpu = gpu
    logger_name = "spice"
    cfg.logger_name = logger_name
    logger = setup_logger(logger_name, cfg.results.output_dir, get_rank())

    # suppress printing if not master
    if cfg.multiprocessing_distributed and cfg.gpu != 0:
        def print_pass(*cfg):
            pass
        builtins.print = print_pass

    if cfg.gpu is not None:
        logger.info("Use GPU: {} for training".format(cfg.gpu))

    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                world_size=cfg.world_size, rank=cfg.rank)

    run_name = "train_self_v2_" + cfg.wandb_run_name
    if cfg.use_wandb and cfg.rank == 0:
        wandb.init(project="domain_dgmc", name=run_name, entity="user", config=vars(cfg))

    columns = ["head1", "head2", "head3", "head4", "head5", "head6", "head7", "head8", "head9", "head10"]

    try:
        main_func(cfg, columns, logger, ngpus_per_node)
    except Exception:
        print(traceback.print_exc(), file=sys.stderr)
    finally:
        if cfg.use_wandb and cfg.rank == 0:
            wandb.finish()

def main_func(cfg, columns, logger, ngpus_per_node):
    df = pd.DataFrame(columns=columns)
    # create model
    model = Sim2Sem(cfg, **cfg.model, gpu=cfg.gpu)
    logger.info(model)
    if cfg.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if cfg.gpu is not None:
            torch.cuda.set_device(cfg.gpu)
            model.cuda(cfg.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
            cfg.workers = int((cfg.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif cfg.gpu is not None:
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    optimizer = make_optimizer(cfg, model)
    scheduler = None
    if "lr_type" in cfg.solver:
        scheduler = make_lr_scheduler(cfg, optimizer)
    # optionally resume from a checkpoint
    if cfg.model.pretrained is not None:
        load_model_weights(model, cfg.model.pretrained, cfg.model.model_type)
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            logger.info("=> loading checkpoint '{}'".format(cfg.resume))
            if cfg.gpu is None:
                checkpoint = torch.load(cfg.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(cfg.gpu)
                checkpoint = torch.load(cfg.resume, map_location=loc)
            cfg.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(cfg.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(cfg.resume))
    # Load similarity model
    cudnn.benchmark = False
    # Data loading code
    train_dataset = build_dataset(cfg.data_train, balance_datasets=cfg.balance_train_self_domains, soft_balance=cfg.soft_balance)
    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    chosen_num_worker = choose_num_workers(generator, train_dataset, train_sampler, cfg.target_sub_batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.target_sub_batch_size, shuffle=(train_sampler is None),
        num_workers=chosen_num_worker, pin_memory=True, sampler=train_sampler, drop_last=True, generator=generator,
        persistent_workers=True)
    dataset_val = build_dataset(cfg.data_test)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=cfg.batch_size_test, shuffle=False,
                                             num_workers=chosen_num_worker, generator=generator,
                                             persistent_workers=True)

    if cfg.style_transfer:
        adain_model = AdaIN(
            "adain_weights/decoder.pth",
            "adain_weights/vgg_normalised.pth",
            cfg.gpu,
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
        )
        adain_model.vgg.eval()
        adain_model.decoder.eval()
        if cfg.use_edges:
            edge_model = ToEdges()
        else:
            edge_model = None
    else:
        adain_model = None
        edge_model=None

    min_loss = 1e10
    best_loss_acc = -2
    best_loss_nmi = -1
    best_loss_ari = -1
    best_loss_chosen_ind = -1
    eval_ent = cfg.eval_ent
    eval_ent_weight = cfg.eval_ent_weight
    all_losses_for_plot = []
    all_accuracies_for_plot = []
    keys = ["head1", "head2", "head3", "head4", "head5", "head6", "head7", "head8", "head9", "head10"]
    for epoch in tqdm(range(cfg.start_epoch, cfg.epochs)):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
        if scheduler is not None:
            scheduler.step()

        # train for one epoch
        train(train_loader, model, optimizer, epoch, cfg,adain_model,edge_model,select_samples_v2=cfg.select_samples_v2)

        if not cfg.multiprocessing_distributed or (cfg.multiprocessing_distributed
                                                   and cfg.rank % ngpus_per_node == 0 and (
                                                           epoch + 1) % cfg.test_freq == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_last.pth.tar'.format(cfg.results.output_dir))
            if (epoch + 1) == cfg.epochs:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename='{}/checkpoint_final.pth.tar'.format(cfg.results.output_dir))
            model.eval()

            loss_fn = nn.CrossEntropyLoss()
            num_heads = len(cfg.model.head.multi_heads)
            gt_labels = []
            pred_labels = []
            scores_all = []
            accs = []
            aris = []
            nmis = []
            feas_sim = []
            domain_labels_all = []
            for h in range(num_heads):
                pred_labels.append([])
                scores_all.append([])

            for ind, (images, _, embs, labels, domain_labels, sample_idx) in enumerate(val_loader):
                images = torch.cat(images).to(cfg.gpu, non_blocking=False)
                with torch.no_grad():
                    if cfg.use_edges:
                        same_domain_photos = edge_model(images)
                        scores = model(same_domain_photos, forward_type="sem")
                    else:
                        scores = model(images, forward_type="sem")

                domain_labels_all.append(torch.cat(domain_labels))
                feas_sim.append(torch.cat(embs))

                assert len(scores) == num_heads
                for h in range(num_heads):
                    pred_idx = scores[h].argmax(dim=1)
                    pred_labels[h].append(pred_idx)
                    scores_all[h].append(scores[h])

                gt_labels.append(torch.cat(labels))

            domain_labels_all = torch.cat(domain_labels_all).long().cpu().numpy()
            gt_labels = torch.cat(gt_labels).long().cpu().numpy()
            feas_sim = torch.cat(feas_sim, dim=0).to(cfg.gpu, non_blocking=False)
            losses = []

            for h in range(num_heads):
                scores_all[h] = torch.cat(scores_all[h], dim=0)
                pred_labels[h] = torch.cat(pred_labels[h], dim=0)

            idx_select, gt_cluster_labels = model(feas_sim=feas_sim, scores=scores_all, epoch=epoch,
                                                  forward_type="sim2sem",
                                                  equalize_domain_sample_select=cfg.equalize_domain_sample_select,
                                                  domain_labels=domain_labels_all,targets_list=gt_labels)
            row_list = []
            length_of_not_chosen_classes = []
            all_classes_selected = []
            for h in range(num_heads):
                pred_labels_h = pred_labels[h].long().cpu().numpy()

                pred_scores_select = scores_all[h][idx_select[h].cpu()]
                gt_labels_select = gt_cluster_labels[h]
                loss = loss_fn(pred_scores_select.cpu(), gt_labels_select)

                if eval_ent:
                    probs = scores_all[h].mean(dim=0)
                    probs = torch.clamp(probs, min=1e-8)
                    ent = -(probs * torch.log(probs)).sum()
                    loss = loss - eval_ent_weight * ent

                acc = calculate_acc(pred_labels_h, gt_labels)  ##not_chosen_classes
                ##check if head predicts all classes in each domain
                for domain in range(len(cfg.data_train.domain_names.split("_"))):
                    curr_inds = domain_labels_all == domain
                    acc = calculate_acc(pred_labels_h[curr_inds], gt_labels[curr_inds])  ##not_chosen_classes
                    if not isinstance(acc,float):
                        all_classes_selected.append(0)
                        break
                if len(all_classes_selected)==h:
                    all_classes_selected.append(1)

                if cfg.keep_strong_heads and epoch >= cfg.epoch2remove_weak_heads and epoch % cfg.remove_weak_heads_every_n_epochs == 0:
                    if not isinstance(acc, float):
                        not_chosen_classes = acc
                        length_of_not_chosen_classes.append(len(not_chosen_classes))
                        not_chosen_classes = np.array2string(acc)
                        acc = NO_ACCURACY
                    else:
                        not_chosen_classes = "all chosen"
                        length_of_not_chosen_classes.append(0)
                else:
                    if not isinstance(acc, float):
                        not_chosen_classes = np.array2string(acc)  # just for clarity
                        acc = NO_ACCURACY
                    else:
                        not_chosen_classes = "all chosen"

                row_list.append(not_chosen_classes)
                print(f"Head:{h},Acc:{round(acc,3)},Loss:{round(loss.item(),3)}")
                nmi = calculate_nmi(pred_labels_h, gt_labels)
                ari = calculate_ari(pred_labels_h, gt_labels)
                accs.append(acc)
                nmis.append(nmi)
                aris.append(ari)
                losses.append(loss.item())

            df.loc[len(df)] = row_list
            tbl = wandb.Table(columns=columns, data=df)

            accs = np.array(accs)
            nmis = np.array(nmis)
            aris = np.array(aris)
            losses = np.array(losses)

            ########################
            all_losses_for_plot.append(losses.tolist())
            all_accuracies_for_plot.append(accs.tolist())
            ########################
            best_acc_real = accs.max()

            losses_arg_sort = np.argsort(losses)
            losses_sort = np.sort(losses)  # low2high

            for index, acc in enumerate(accs[losses_arg_sort]):
                if acc != NO_ACCURACY and (all_classes_selected[index] or sum(all_classes_selected)==0):
                    chosen_head_loss = losses_sort[index]
                    chosen_head_index = losses_arg_sort[index]
                    break
            else:
                chosen_head_loss = 10e10
                chosen_head_index = losses_arg_sort[0]

            if min_loss > chosen_head_loss:
                min_loss = chosen_head_loss  # losses.min()
                best_loss_acc = accs[chosen_head_index]
                best_loss_nmi = nmis[chosen_head_index]
                best_loss_ari = aris[chosen_head_index]
                best_loss_chosen_ind = chosen_head_index

                state_dict = model.state_dict()
                state_dict_save = {}
                for k in list(state_dict.keys()):
                    if not k.startswith('module.head'):
                        state_dict_save[k] = state_dict[k]

                    if k.startswith('module.head.head_{}'.format(best_loss_chosen_ind)):
                        state_dict_save['module.head.head_0.{}'.format(
                            k[len('module.head.head_{}.'.format(best_loss_chosen_ind))::])] = state_dict[k]
                torch.save(state_dict_save, '{}/checkpoint_select.pth.tar'.format(cfg.results.output_dir))

            if cfg.keep_strong_heads and epoch >= cfg.epoch2remove_weak_heads and epoch % cfg.remove_weak_heads_every_n_epochs == 0:
                length_np = np.array(length_of_not_chosen_classes)
                lowest_lengths = length_np[np.argsort(length_np)[:2].tolist()]
                lowest_lengths = np.unique(lowest_lengths)
                heads2keep = []
                for val in lowest_lengths:
                    indexes_that_are_equal_to_this_low_number_of_cls_chosen = (length_np == val).nonzero()[
                        0]
                    for p in indexes_that_are_equal_to_this_low_number_of_cls_chosen:
                        heads2keep.append(p)
                        if len(heads2keep)==cfg.heads2keep:
                            break
                keep_the_strong_heads_func(model, heads2keep)

            model.train()

            if cfg.use_wandb:
                temp = np.array(all_accuracies_for_plot).transpose()
                plot_for_wandb_acc = wandb.plot.line_series(
                    xs=range(0, len(all_accuracies_for_plot)),
                    ys=temp,
                    keys=keys,
                    title="Acc for all heads",
                    xname="Epochs")
                temp1 = np.array(all_losses_for_plot).transpose()
                plot_for_wandb_loss = wandb.plot.line_series(
                    xs=range(0, len(all_losses_for_plot)),
                    ys=temp1,
                    keys=keys,
                    title="Loss for all heads",
                    xname="Epochs")
                wandb.log(
                    {"non chosen table": tbl, "select acc": best_loss_acc, "select nmi": best_loss_nmi,
                     "select ari": best_loss_ari, "select chosen_head_index": best_loss_chosen_ind,
                     "current best_acc": best_acc_real, "min_loss": min_loss, "current_head_loss": chosen_head_loss,
                     "lr": optimizer.param_groups[0]['lr'], "AccAllHeads": plot_for_wandb_acc,
                     "AllHeadsLoss": plot_for_wandb_loss})
            else:
                logger.info(f"non chosen table: {tbl}, select acc: {best_loss_acc}, select nmi: {best_loss_nmi},"
                            f"select ari: {best_loss_ari}, select chosen_head_index: {best_loss_chosen_ind},"
                            f"current best_acc: {best_acc_real}, min_loss: {min_loss}, current_head_loss: {chosen_head_loss},"
                            f"lr: {optimizer.param_groups[0]['lr']}")
    if cfg.use_wandb:
        wandb.finish()


def train(train_loader, model, optimizer, epoch, cfg,adain_model=None,edge_model=None,select_samples_v2=False):
    logger = logging.getLogger("{}.trainer".format(cfg.logger_name))

    info = []
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    info.append(batch_time)
    info.append(data_time)
    num_heads = len(cfg.model.head.multi_heads)
    losses = []
    for h in range(num_heads):
        losses_h = AverageMeter('Loss_{}'.format(h), ':.4e')
        losses.append(losses_h)
        info.append(losses_h)

    lr = AverageMeter('lr', ':.6f')
    lr.update(optimizer.param_groups[0]["lr"])
    info.append(lr)

    progress = ProgressMeter(
        len(train_loader),
        info,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    target_sub_batch_size = cfg.solver.target_sub_batch_size
    batch_size = cfg.solver.batch_size
    train_sub_batch_size = cfg.solver.train_sub_batch_size

    num_repeat = cfg.solver.num_repeat

    num_imgs_all = len(train_loader.dataset)

    iters_end = batch_size // target_sub_batch_size
    num_iters_l = num_imgs_all // batch_size
    for ii in range(num_iters_l):
        end = time.time()
        model.eval()
        scores = []
        for h in range(num_heads):
            scores.append([])

        images_trans_l = []
        feas_sim = []
        domain_labels_all = []

        for step, (weak_aug_batch, strong_aug_batch, feas_sim_batch, targets_sem, domain_labels, sample_idx) in enumerate(train_loader):
            if step==0:
                targets_list=targets_sem[0]
            else:
                targets_list = torch.cat([targets_list,targets_sem[0]],dim=0)
            # measure data loading time
            data_time.update(time.time() - end)
            # print(images_ori_l_batch.shape)
            domain_labels_all.append(torch.cat(domain_labels))
            # Generate ground truth.
            ####################
            # Generate style transferred images
            ####################
            strong_aug_batch[0] = strong_aug_batch[0].to(cfg.gpu, non_blocking=False)
            if cfg.style_transfer:
                num_domains = len(cfg.data_train.domain_names.split("_"))
                all_domains = list(range(num_domains))
                style_transfer_images = []
                for domain_idx in all_domains:
                    other_domains = all_domains.copy()
                    other_domains.remove(domain_idx)
                    traget_domain_style = np.random.choice(np.array(other_domains), 1)[0]
                    # Transfer
                    where_eq_to_domain_idx = torch.where(domain_labels[0] == domain_idx)[0]
                    len_of_domain_lbls_input = torch.sum(domain_labels[0] == domain_idx)
                    len_of_domain_lbls_target = torch.sum(domain_labels[0] == traget_domain_style)
                    domain_target_images = weak_aug_batch[0][domain_labels[0] == traget_domain_style].cuda(cfg.gpu,
                                                                                                        non_blocking=True)
                    if len_of_domain_lbls_target==0:
                        continue
                    if len_of_domain_lbls_input < len_of_domain_lbls_target:
                        domain_target_images = domain_target_images[:len_of_domain_lbls_input]
                    elif len_of_domain_lbls_input > len_of_domain_lbls_target:
                        mod    = len_of_domain_lbls_input.item()  % len_of_domain_lbls_target.item()
                        repeat = len_of_domain_lbls_input.item() // len_of_domain_lbls_target.item()
                        domain_target_images =  domain_target_images.repeat(repeat+1,1,1,1)[:-1*(len_of_domain_lbls_target-mod)]
                    if cfg.use_edges and random.random()>0.5:
                        xu_k_sty = edge_model(weak_aug_batch[0][domain_labels[0]==domain_idx].cuda(cfg.gpu, non_blocking=True))
                    else:
                        if random.random()< cfg.p_bcd_augment:
                            sketch_style_images = glob.glob("./style_sketch_images/*")
                            img_idx = np.random.randint(0,len(sketch_style_images))
                            image_arr = torchvision.transforms.Resize((224, 224))(internal_read_image(sketch_style_images[img_idx])).cuda()
                            if image_arr.shape[0]==1:
                              image_arr= image_arr.repeat(3,1,1)
                            image_arr[0] = ((image_arr[0] - torch.min(image_arr[0])) / (
                                    torch.max(image_arr[0]) - torch.min(image_arr[0]) + 10e-10)).cuda() * 2
                            with torch.no_grad():
                                xu_k_sty = adain_model(weak_aug_batch[0][domain_labels[0] == domain_idx].cuda(cfg.gpu, non_blocking=True),
                                                                 image_arr.unsqueeze(dim=0).repeat(weak_aug_batch[0][domain_labels[0] == domain_idx].shape[0], 1, 1, 1))
                        else:
                            with torch.no_grad():
                                xu_k_sty = adain_model(
                                        weak_aug_batch[0][domain_labels[0] == domain_idx].cuda(cfg.gpu, non_blocking=True),
                                        domain_target_images)

                    choice_p_precent_domain = np.random.choice([0, 1], p=[1 - cfg.p_style_transfer, cfg.p_style_transfer],
                                     size=len_of_domain_lbls_input.item())
                    choice_p_precent_absolute = where_eq_to_domain_idx[choice_p_precent_domain==1]

                    strong_aug_batch[0][choice_p_precent_absolute] = xu_k_sty[choice_p_precent_domain==1]

            # Select samples and estimate the ground-truth relationship between samples.
            weak_aug_batch = torch.cat(weak_aug_batch).to(cfg.gpu, non_blocking=False)
            with torch.no_grad():
                if cfg.style_transfer and random.random() > cfg.p_suprise_bcd:
                    sketch_style_images = glob.glob("./style_sketch_images/*")
                    image_arr = [torchvision.transforms.Resize((224, 224))(internal_read_image(img)).cuda() for img in
                                 sketch_style_images]
                    image_arr[0] = ((image_arr[0] - torch.min(image_arr[0])) / (
                                torch.max(image_arr[0]) - torch.min(image_arr[0]) + 10e-10)).cuda() * 2
                    same_domain_photos = adain_model(weak_aug_batch,
                         image_arr[0].unsqueeze(dim=0).repeat(weak_aug_batch.shape[0], 1, 1, 1))
                    scores_nl = model(same_domain_photos, forward_type="sem")
                elif cfg.use_edges:
                    same_domain_photos = edge_model(weak_aug_batch)#,
                                      #weak_aug_batch[domain_labels[0] == 2][0].unsqueeze(dim=0).repeat(weak_aug_batch.shape[0], 1, 1, 1))
                    scores_nl = model(same_domain_photos, forward_type="sem")
                else:
                    scores_nl = model(weak_aug_batch, forward_type="sem")
            assert num_heads == len(scores_nl)

            for h in range(num_heads):
                scores[h].append(scores_nl[h].detach())

            images_trans_l.append(torch.cat(strong_aug_batch))
            feas_sim.append(torch.cat(feas_sim_batch))

            if len(feas_sim) >= iters_end:
                train_loader.sampler.set_epoch(train_loader.sampler.epoch + 1)
                break

        for h in range(num_heads):
            scores[h] = torch.cat(scores[h], dim=0)

        images_trans_l = torch.cat(images_trans_l)
        feas_sim = torch.cat(feas_sim).to(cfg.gpu).to(torch.float32)
        domain_labels_all = torch.cat(domain_labels_all).long().cpu().numpy()

        idx_select, gt_cluster_labels = model(feas_sim=feas_sim, scores=scores, epoch=epoch, forward_type="sim2sem",
                                                  equalize_domain_sample_select=cfg.equalize_domain_sample_select,
                                                  domain_labels=domain_labels_all,targets_list=targets_list,
                                              select_samples_max_k = cfg.select_samples_max_k,select_samples_v2=select_samples_v2)

        images_trans = []
        for h in range(num_heads):
            images_trans.append(images_trans_l[idx_select[h], :, :, :])

        num_imgs = images_trans[0].shape[0]

        # Train with the generated ground truth
        model.train()
        img_idx = list(range(num_imgs))
        # Select a set of images for training.

        num_train = num_imgs
        # train_sub_iters = int(torch.ceil(float(num_train) / train_sub_batch_size))
        train_sub_iters = num_train // train_sub_batch_size

        for n in range(num_repeat):
            random.shuffle(img_idx)

            for i in range(train_sub_iters):
                start_idx = i * train_sub_batch_size
                end_idx = min((i + 1) * train_sub_batch_size, num_train)
                img_idx_i = img_idx[start_idx:end_idx]

                imgs_i = []
                targets_i = []

                for h in range(num_heads):
                    imgs_i.append(images_trans[h][img_idx_i, :, :, :].to(cfg.gpu, non_blocking=False))
                    targets_i.append(gt_cluster_labels[h][img_idx_i].to(cfg.gpu, non_blocking=False))
                loss_dict = model(imgs_i, target=targets_i, forward_type="loss")
                loss = sum(loss for loss in loss_dict.values())#if i % 10==0:
                loss_mean = loss / num_heads
                loss_mean.backward()
                if i % cfg.self_grad_accum==0:
                    optimizer.step()
                    optimizer.zero_grad()

                for h in range(num_heads):
                    # measure accuracy and record loss
                    losses[h].update(loss_dict['head_{}'.format(h)].item(), imgs_i[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ii % cfg.print_freq == 0:
            logger.info(progress.display(ii))


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
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
