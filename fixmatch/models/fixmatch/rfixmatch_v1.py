import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler

import os, sys
import contextlib
from fixmatch.train_utils import AverageMeter
from spice.model.heads.domain_head import DomainClassifierReverse
from .fixmatch_utils import consistency_loss, Get_Scalar
from fixmatch.train_utils import ce_loss
import numpy as np
from scipy.optimize import linear_sum_assignment
from spice.utils.evaluation import calculate_acc, calculate_nmi, calculate_ari
from moco.builder import FFN
import wandb
import traceback
from adain.adain import AdaIN


class FixMatch:
    def __init__(self, args, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, \
                 hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None,
                 feature_dim=512, domain_loss_weight=0.1, gpu=0, use_domain_classifier=True, mlp=True,
                 domain_names=None):
        """
        class Fixmatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes 
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(FixMatch, self).__init__()
        self.args = args
        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py

        self.train_model = net_builder(num_classes=num_classes)
        self.eval_model = net_builder(num_classes=num_classes)

        if self.args.style_transfer:
            self.adain_model = AdaIN(
                "adain_weights/decoder.pth",
                "adain_weights/vgg_normalised.pth",
                args.gpu,
                norm_mean=[0.485, 0.456, 0.406],
                norm_std=[0.229, 0.224, 0.225],
            )

        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        self.it = 0

        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.train_model.fc.weight.shape[1]
            self.train_model.fc = FFN(
                encoder_q_out_dim=self.num_classes)  # nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.eval_model.fc = FFN(
                encoder_q_out_dim=self.num_classes)  # nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net

        self.eval_model.eval()

        self.use_domain_classifier = use_domain_classifier

        if use_domain_classifier:
            self.num_domains = len(domain_names.split("_"))
            self.domain_head = DomainClassifierReverse(self.args.domain_size_layers, feature_dim, domain_loss_weight,
                                                       gpu,
                                                       num_domains=self.num_domains)

    @torch.no_grad()
    def _eval_model_update(self):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        for param_train, param_eval in zip(self.train_model.module.parameters(), self.eval_model.parameters()):
            param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1 - self.ema_m))

        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args, epoch, logger=None):
        """
        Train function of FixMatch.
        From data_loader, it inference training data, computes losses, and update the networks.
        """
        ngpus_per_node = torch.cuda.device_count()

        # lb: labeled, ulb: unlabeled
        run_name = "train_semi_" + args.wandb_run_name
        if args.use_wandb and args.rank == 0:
            wandb.init(project="domain_dgmc", name=run_name, entity="user", config=vars(cfg))
        try:
            eval_dict = self.main_func(args, epoch, ngpus_per_node)
        except Exception:
            print(traceback.print_exc(), file=sys.stderr)
        finally:
            if args.use_wandb and args.rank == 0:
                wandb.finish()

        return eval_dict

    def main_func(self, args, epoch, ngpus_per_node):
        self.train_model.train()
        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        start_batch.record()
        best_eval_acc, best_it = 0, 0  # [0.0] * self.num_domains, 0
        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext
        for step, (
                (x_lb, y_lb, idx, domain_label_lb, sample_idx),
                (x_ulb_w, x_ulb_s, _, domain_label_ulb, sample_idx_ulb)) in \
                enumerate(zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb'])):

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break

            ####################
            # Generate style transferred images
            ####################
            if args.style_transfer:
                domains_in_ubatch = list(torch.unique(domain_label_ulb))
                style_transfer_images = []
                chosen_style_ind = torch.randperm(x_ulb_w.size()[0])

                # Transfer
                x_ulb_sty = self.adain_model(x_ulb_w.cuda(args.gpu, non_blocking=True),
                                             x_ulb_w[chosen_style_ind].cuda(args.gpu, non_blocking=True))

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)
            if args.style_transfer:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s, x_ulb_sty))
            else:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

            # inference and calculate sup/unsup losses
            with amp_cm():
                logits, emb_q = self.train_model(inputs)
                logits_x_lb = logits[:num_lb]

                if args.style_transfer:
                    logits_x_ulb_w, logits_x_ulb_s, logits_x_ulb_sty = logits[num_lb:].chunk(3)
                else:
                    logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                    logits_x_ulb_sty = None

                del logits

                # hyper-params for update
                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)

                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                unsup_loss, mask = consistency_loss(logits_x_ulb_w,
                                                    logits_x_ulb_s,
                                                    'ce', T, p_cutoff,
                                                    use_hard_labels=args.hard_label)

                if args.style_transfer:
                    style_loss, mask = consistency_loss(logits_x_ulb_w,
                                                        logits_x_ulb_sty,
                                                        'ce', T, p_cutoff,
                                                        use_hard_labels=args.hard_label)
                else:
                    style_loss = 0
                if self.use_domain_classifier:
                    domain_labels = torch.cat((domain_label_lb, domain_label_ulb, domain_label_ulb))
                    domain_labels = domain_labels.cuda(args.gpu, non_blocking=False)
                    p = float(step + epoch * len(self.loader_dict['train_lb'])) / args.epoch / len(
                        self.loader_dict['train_lb'])
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                else:
                    domain_labels = None
                    alpha = 0

                if self.use_domain_classifier:
                    if args.style_transfer:
                        domain_loss = self.domain_head.loss(emb_q[:-len(logits_x_ulb_sty)], domain_labels, alpha)
                    else:
                        domain_loss = self.domain_head.loss(emb_q, domain_labels, alpha)
                else:
                    domain_loss = 0

                total_loss = sup_loss + self.lambda_u * unsup_loss + domain_loss + style_loss

            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.train_model.zero_grad()

            with torch.no_grad():
                self._eval_model_update()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach()
            tb_dict['train/unsup_loss'] = unsup_loss.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['train/mask_ratio'] = 1.0 - mask.detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            if self.it % self.num_eval_iter == 0:

                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)

                save_path = os.path.join(args.save_dir, args.save_name)

                if np.mean(tb_dict['eval/top-1-acc']) > np.mean(best_eval_acc):
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it
                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {hasattr(self, 'eval_model')}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")

                if args.use_wandb:
                    wandb.log({"sup_loss": sup_loss.detach(), "unsup_loss": unsup_loss.detach(),
                               "total_loss": total_loss.detach(), "domain_loss": domain_loss,
                               "eval-top-1": eval_dict['eval/top-1-acc'], "eval-nmi": eval_dict['eval/nmi'],
                               "eval-ari": eval_dict['eval/ari'], "best_eval_acc": best_eval_acc,"lr": tb_dict['lr']})



            elif self.it % (self.num_eval_iter // 10) == 0:
                self.print_fn(f"{self.it} iteration,  {tb_dict}")
                if args.use_wandb:
                    wandb.log({"sup_loss": sup_loss.detach(), "unsup_loss": unsup_loss.detach(),
                               "total_loss": total_loss.detach(), "domain_loss": domain_loss})

            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                if self.it % self.num_eval_iter == 0:
                    self.save_model('model_last.pth', save_path)

                if self.it % self.num_eval_iter == 0:
                    self.save_model('model_{}.pth'.format(self.it), save_path)

                if self.it == best_it:
                    self.save_model('model_best.pth', save_path)

                # if not self.tb_log is None:
                # self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            start_batch.record()
            if self.it > 2 ** 19:
                self.num_eval_iter = 1000
        if args.use_wandb:
            wandb.finish()
        eval_dict = self.evaluate(args=args, per_domain=True)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None, per_domain=False):
        use_ema = hasattr(self, 'eval_model')

        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.eval()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']

        labels_pred = []
        labels_gt = []
        domain_labels_gt = []
        for x, y, idx, domain_lbl, sample_idx in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            logits, _ = eval_model(x)

            labels_pred.append(torch.max(logits, dim=-1)[1].cpu().numpy())
            labels_gt.append(y.cpu().numpy())
            domain_labels_gt.append(domain_lbl.cpu().numpy())

        labels_pred = np.concatenate(labels_pred, axis=0)
        labels_gt = np.concatenate(labels_gt, axis=0)
        domain_labels_gt = np.concatenate(domain_labels_gt, axis=0)

        acc, ari, nmi = FixMatch.calc_metrics(labels_gt, labels_pred)
        if not isinstance(acc, float):
            acc = -1

        if per_domain:
            acc_per_domain, ari_per_domain, nmi_per_domain = [], [], []
            for domain in range(self.num_domains):
                curr_inds = domain_labels_gt == domain
                curr_acc, curr_ari, curr_nmi = FixMatch.calc_metrics(labels_gt[curr_inds], labels_pred[curr_inds])
                acc_per_domain.append(curr_acc)
                ari_per_domain.append(curr_ari)
                nmi_per_domain.append(curr_nmi)
            print("Acc per domain:", acc_per_domain, "Ari per domain:", ari_per_domain, "Nmi per domain:",
                  nmi_per_domain)
        if not use_ema:
            eval_model.train()

        return {'eval/loss': -1, 'eval/top-1-acc': acc, 'eval/nmi': nmi, 'eval/ari': ari}

    @staticmethod
    def calc_metrics(labels_gt, labels_pred):
        try:
            acc = calculate_acc(labels_pred, labels_gt)
        except:
            acc = -1
        nmi = calculate_nmi(labels_pred, labels_gt)
        ari = calculate_ari(labels_pred, labels_gt)
        return acc, ari, nmi

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        torch.save({'train_model': train_model.state_dict(),
                    'eval_model': eval_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it}, save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def remove_feature_hirerchy(self, dict):
        from collections import OrderedDict
        train_dict = OrderedDict()
        eval_dict = OrderedDict()
        domain_dict = OrderedDict()
        for k, v in dict.items():
            if "feature_module" in k:
                new_k = k.replace("feature_module.", "")
                new_k_eval = k.replace("module.feature_module.", "")
            else:
                new_k = k
                new_k_eval = k
            if "classifier" in new_k and "lin" in new_k:
                if "classifier.lin1.weight" in new_k:
                    new_k = new_k.replace("module.head.head_0.classifier.lin1.weight", "module.fc.fc1.weight")
                    new_k_eval = new_k_eval.replace("module.head.head_0.classifier.lin1.weight", "fc.fc1.weight")

                elif "classifier.lin1.bias" in new_k:
                    new_k = new_k.replace("module.head.head_0.classifier.lin1.bias", "module.fc.fc1.bias")
                    new_k_eval = new_k_eval.replace("module.head.head_0.classifier.lin1.bias", "fc.fc1.bias")
                elif "classifier.lin2.weight" in new_k:
                    new_k = new_k.replace("module.head.head_0.classifier.lin2.weight", "module.fc.fc2.weight")
                    new_k_eval = new_k_eval.replace("module.head.head_0.classifier.lin2.weight", "fc.fc2.weight")
                elif "classifier.lin2.bias" in new_k:
                    new_k = new_k.replace("module.head.head_0.classifier.lin2.bias", "module.fc.fc2.bias")
                    new_k_eval = new_k_eval.replace("module.head.head_0.classifier.lin2.bias", "fc.fc2.bias")

            if "domain" in new_k:
                new_k = new_k.replace("module.domain_head.", "")

                domain_dict[new_k] = v
            else:
                train_dict[new_k] = v
                eval_dict[new_k_eval] = v
        return train_dict, eval_dict, domain_dict

    def load_model(self, load_path, resume=False):
        checkpoint = torch.load(load_path, map_location="cpu")
        if resume:
            train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
            eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model

            for key in checkpoint.keys():
                if hasattr(self, key) and getattr(self, key) is not None:
                    if 'train_model' in key:
                        train_model.load_state_dict(checkpoint[key])
                    elif 'eval_model' in key:
                        eval_model.load_state_dict(checkpoint[key])
                    elif key == 'it':
                        self.it = checkpoint[key]
                    else:
                        getattr(self, key).load_state_dict(checkpoint[key])
                    self.print_fn(f"Check Point Loading: {key} is LOADED")
                else:
                    self.print_fn(f"Check Point Loading: {key} is **NOT** LOADED")
        else:
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            checkpoint, eval_dict, domain_checkpoint = self.remove_feature_hirerchy(checkpoint)
            msg  = self.train_model.load_state_dict(checkpoint,strict=False)
            print(msg)
            self.eval_model.load_state_dict(eval_dict,strict=False)
            moco_load_path = load_path.replace("spice_self","moco")

            dict = torch.load(moco_load_path)["state_dict"]
            from collections import OrderedDict
            new_dict = OrderedDict()
            for k,v in dict.items():
                if "domain_classifier" in k:
                    new_k = k.replace("module.domain_head.","")
                    new_dict[new_k]=v
            self.domain_head.load_state_dict(new_dict)


if __name__ == "__main__":
    pass
