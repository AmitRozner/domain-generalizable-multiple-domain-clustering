import torch
import torch.nn as nn
from ..feature_modules.build_feature_module import build_feature_module

import numpy as np
import torch.nn.functional as F
from torch.nn import functional as F


def ce_smooth(input, target, label_smoothing_val=0.0, pred_based_hist=None):
    n_cls = input.shape[-1]
    if pred_based_hist is not None:
        smooth_per_sample = 1 - pred_based_hist[torch.argmax(input, dim=1).cpu()]
        smooth_per_sample = torch.clip(smooth_per_sample, min=label_smoothing_val).unsqueeze(dim=1)
    else:
        smooth_per_sample = label_smoothing_val

    smooth = (1 - smooth_per_sample) / (n_cls - 1)
    target_oh = F.one_hot(target.cpu(), n_cls)
    if pred_based_hist is not None:
        target_oh = target_oh + smooth
        for row in range(len(smooth)):
            target_oh[row, :] = torch.clip(target_oh[row, :], min=0, max=(1 - (n_cls * smooth[row, :])).cpu().item())
    else:
        target_oh = torch.clip(torch.add(target_oh, smooth), min=0, max=1 - (n_cls * smooth))

    log_probs = torch.nn.functional.log_softmax(input, dim=1).cpu()
    return -(target_oh * log_probs).sum() / input.shape[0]
class SemHead(nn.Module):
    def __init__(self, classifier, feature_conv=None, num_cluster=10, center_ratio=0.5,
                 iter_start=0, iter_up=-1, iter_down=-1, iter_end=0, ratio_start=0.5, ratio_end=0.95, loss_weight=None,
                 fea_fc=False, T=1, sim_ratio=1, sim_center_ratio=0.9, epoch_merge=5, entropy=False,label_smoothing=0.0):

        super(SemHead, self).__init__()
        if loss_weight is None:
            loss_weight = dict(loss_cls=1, loss_ent=0)
        self.loss_weight = loss_weight
        self.classifier = build_feature_module(classifier)
        self.feature_conv = None
        if feature_conv:
            self.feature_conv = build_feature_module(feature_conv)
        self.label_smoothing = label_smoothing
        self.num_cluster = num_cluster
        self.loss_fn_cls = nn.CrossEntropyLoss()
        self.iter_start = iter_start
        self.iter_end = iter_end
        self.ratio_start = ratio_start
        self.ratio_end = ratio_end
        self.center_ratio = center_ratio
        self.ave_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.fea_fc = fea_fc
        self.T = T
        self.sim_ratio = sim_ratio
        self.iter_up = iter_up
        self.iter_down = iter_down
        self.sim_center_ratio = sim_center_ratio
        self.epoch_merge = epoch_merge

        self.entropy = entropy
        self.EPS = 1e-5

    def compute_ratio_selection_old(self, i):
        if self.ratio_end == self.ratio_start:
            return self.ratio_start
        elif self.iter_start < i <= self.iter_end:
            r = (self.ratio_end - self.ratio_start) / (self.iter_end - self.iter_start) * (i - self.iter_start) + self.ratio_start
            return r
        else:
            return self.ratio_start

    def compute_ratio_selection(self, i):
        if self.ratio_end == self.ratio_start:
            return self.ratio_start
        elif self.iter_up != -1 and self.iter_down != -1:
            if i < self.iter_start:
                return self.ratio_start
            elif self.iter_start <= i < self.iter_up:
                r = (self.ratio_end - self.ratio_start) / (self.iter_up - self.iter_start) * (i - self.iter_start) + self.ratio_start
                return r
            elif self.iter_up <= i < self.iter_down:
                return self.ratio_end
            elif self.iter_down <= i < self.iter_end:
                r = self.ratio_end - (self.ratio_end - self.ratio_start) / (self.iter_end - self.iter_down) * (i - self.iter_down)
                return r
            else:
                return self.ratio_start
        else:
            if self.iter_start < i <= self.iter_end:
                r = (self.ratio_end - self.ratio_start) / (self.iter_end - self.iter_start) * (i - self.iter_start) + self.ratio_start
                return r
            else:
                return self.ratio_start

    def select_samples_cpu(self, feas_sim, scores, i):

        _, idx_max = torch.sort(scores, dim=0, descending=True)
        idx_max = idx_max.cpu()
        num_per_cluster = idx_max.shape[0] // self.num_cluster
        ratio_select = self.compute_ratio_selection(i)
        k = int(self.center_ratio * num_per_cluster * ratio_select)
        print(k)
        idx_max = idx_max[0:k, :]

        centers = []
        for c in range(self.num_cluster):
            centers.append(feas_sim[idx_max[:, c], :].mean(axis=0))

        select_idx_all = []
        select_labels_all = []
        num_per_cluster = feas_sim.shape[0] // self.num_cluster
        ratio_select = self.compute_ratio_selection(i)
        num_select_c = int(num_per_cluster * ratio_select)
        for c in range(self.num_cluster):
            center_c = centers[c]
            dis = np.dot(feas_sim, center_c.T).squeeze()
            idx_s = np.argsort(dis)[::-1]
            idx_select = idx_s[0:num_select_c]

            select_idx_all = select_idx_all + list(idx_select)
            select_labels_all = select_labels_all + [c] * len(idx_select)

        select_idx_all = np.array(select_idx_all)
        select_labels_all = np.array(select_labels_all)

        return select_idx_all, select_labels_all

    def select_samples(self, feas_sim, scores, epoch,last_idx_max,targets_list,domain_labels,args=None):
        _, idx_max = torch.sort(scores, dim=0, descending=True)
        idx_max = idx_max.cpu()
        num_per_cluster = idx_max.shape[0] // self.num_cluster
        ratio_select = self.compute_ratio_selection(epoch)
        # print(ratio_select)
        k = int(self.center_ratio * num_per_cluster * ratio_select)
        # print(k, len(idx_max))
        idx_max = idx_max[0:k, :]

        centers = []
        for c in range(self.num_cluster):
            centers.append(feas_sim[idx_max[:, c], :].mean(axis=0).unsqueeze(dim=0))

        centers = torch.cat(centers, dim=0)
        torch.save(centers, "/".join(args["pre_model"].split("/")[:-2]) + "/spice_self/centers.pth")
        num_select_c = int(num_per_cluster * ratio_select)

        dis = torch.einsum('cd,nd->cn', [centers / torch.norm(centers, dim=1).unsqueeze(-1), feas_sim])
        idx_select = torch.argsort(dis, dim=1, descending=True)[:, 0:num_select_c].flatten()
        labels_select = torch.arange(0, self.num_cluster).unsqueeze(dim=1).repeat(1, num_select_c).flatten()
        return idx_select, labels_select,idx_max

    def select_samples_domain(self, feas_sim, scores, epoch, idx_max_list, targets_list, domain_labels, select_samples_max_k=10):
        _, idx_max = torch.sort(scores, dim=0, descending=True)
        idx_max = idx_max.cpu()
        num_examples,num_classes=idx_max.shape
        num_per_cluster = idx_max.shape[0] // self.num_cluster
        ratio_select = self.compute_ratio_selection(epoch)
        k = min(select_samples_max_k, max(1, int(self.center_ratio * num_per_cluster * ratio_select)))
        num_domains = len(np.unique(domain_labels))
        ratio_select = self.compute_ratio_selection(epoch)
        k = min(select_samples_max_k, max(num_domains, int(self.center_ratio * num_per_cluster * ratio_select)))
        idx_sorted_domain_lbls = domain_labels[idx_max]
        rest_idx_sorted_domain_lbls = idx_sorted_domain_lbls[2:,:]
        domain_idx_max = idx_max[:2,:]
        rest_idx_max = idx_max[2:, :]
        choose_per_domain = (k // num_domains)#+1
        for ind, domain in enumerate(np.unique(domain_labels)):
            choosen_idx_max_that_has_domain_lbl = []
            for cls_idx in range(num_classes):
                idx_max_that_has_domain_lbl = rest_idx_max[:,cls_idx][rest_idx_sorted_domain_lbls[:,cls_idx] == domain]
                choosen_idx_max_that_has_domain_lbl.append(idx_max_that_has_domain_lbl[:choose_per_domain])
            cur_domain_samples = torch.cat(choosen_idx_max_that_has_domain_lbl).view(choose_per_domain,num_classes)
            domain_idx_max = torch.cat((domain_idx_max,cur_domain_samples))
        centers = []
        for c in range(self.num_cluster):
            centers.append(feas_sim[domain_idx_max[:, c], :].mean(axis=0).unsqueeze(dim=0))

        centers = torch.cat(centers, dim=0)

        num_select_c = int(num_per_cluster * ratio_select)
        if num_select_c%num_domains ==0:
            choose_without_relate_to_domain = num_domains
            choose_in_domain = num_select_c//num_domains-1
        else:
            choose_without_relate_to_domain = num_select_c%num_domains
            choose_in_domain = num_select_c // num_domains
        dis = torch.einsum('cd,nd->cn', [centers / torch.norm(centers, dim=1).unsqueeze(-1), feas_sim])
        idx_select = torch.argsort(dis, dim=1, descending=True)[:, 0:num_select_c].flatten()
        labels_select = torch.arange(0, self.num_cluster).unsqueeze(dim=1).repeat(1, num_select_c).flatten()
        # if num_select_c%num_domains ==0:
        #     choose_without_relate_to_domain = num_domains
        #     choose_in_domain = num_select_c//num_domains-1
        # else:
        #     choose_without_relate_to_domain = num_select_c%num_domains
        #     choose_in_domain = num_select_c // num_domains
        # dis = torch.einsum('cd,nd->cn', [centers / torch.norm(centers, dim=1).unsqueeze(-1), feas_sim])
        # idx_select = torch.argsort(dis, dim=1, descending=True).cpu()[:,:choose_without_relate_to_domain]
        # for dom in range(num_domains):
        #     cur = torch.argsort(dis, dim=1, descending=True).cpu()[:, choose_without_relate_to_domain:][:, idx_sorted_domain_lbls[choose_without_relate_to_domain:] == dom][:, :choose_in_domain]
        #     idx_select = torch.cat([idx_select,cur],dim=-1)
        # idx_select  = idx_select.flatten()
        # labels_select = torch.arange(0, self.num_cluster).unsqueeze(dim=1).repeat(1, num_select_c).flatten()
        return idx_select, labels_select,idx_max

    def select_samples_v2(self, feas_sim, scores, i, args=None):

        _, idx_max = torch.sort(scores, dim=0, descending=True)
        idx_max = idx_max.cpu()
        num_per_cluster = idx_max.shape[0] // self.num_cluster
        ratio_select = self.compute_ratio_selection(i)
        # print(ratio_select)
        k = int(self.center_ratio * num_per_cluster * ratio_select)
        # print(k, len(idx_max))

        idx_center_exist = torch.zeros_like(idx_max[:, 0], dtype=torch.bool)

        centers = []
        for c in np.random.permutation(self.num_cluster):
            idx_c = idx_max[:, c]
            if c == 0:
                idx_c_select = idx_c[0:k]
            else:
                idx_c_available = ~idx_center_exist[idx_c]
                idx_c_select = idx_c[idx_c_available][0:k]

            idx_center_exist[idx_c_select] = True

            centers.append(feas_sim[idx_c_select, :].mean(axis=0).unsqueeze(dim=0))

        centers = torch.cat(centers, dim=0)
        torch.save(centers, "/".join(args["pre_model"].split("/")[:-2]) + "/spice_self/centers.pth")

        num_select_c = int(num_per_cluster * ratio_select)

        dis = torch.einsum('cd,nd->cn', [centers / torch.norm(centers, dim=1).unsqueeze(-1), feas_sim])
        # idx_select = torch.argsort(dis, dim=1, descending=True)[:, 0:num_select_c].flatten()
        ##TODO sample_select_v1 try to select here as in the above comment and remove the following
        idx_sort = torch.argsort(dis, dim=1, descending=True)
        idx_label_exist = torch.zeros_like(idx_sort[0, :], dtype=torch.bool)
        labels_select_all = []
        idx_select_all = []
        for c in np.random.permutation(self.num_cluster):
            idx_c = idx_sort[c, :]
            if c == 0:
                idx_c_select = idx_sort[0, 0:num_select_c]
            else:
                idx_c_available = ~idx_label_exist[idx_c]
                idx_c_select = idx_c[idx_c_available][0:num_select_c]

            idx_label_exist[idx_c_select] = True
            idx_select_all.append(idx_c_select)
            labels_select_all.append(torch.zeros_like(idx_c_select)+c)

        idx_select_all = torch.cat(idx_select_all)
        labels_select_all = torch.cat(labels_select_all)
        print(len(set(idx_select_all.cpu().numpy())))

        return idx_select_all, labels_select_all, idx_max

    def forward(self, fea, **kwargs):

        if self.feature_conv is not None:
            fea_conv = self.feature_conv(fea)
        else:
            fea_conv = fea

        if not self.fea_fc:
            feature = self.ave_pooling(fea_conv)
            feature = feature.flatten(start_dim=1)
        else:
            feature = fea_conv

        cls_score = self.classifier(feature)

        cls_score = cls_score / self.T

        return cls_score

    def loss(self, x, target, pred_based_hist=None, **kwargs):
        cls_score = self.forward(x)
        curr_hist = torch.histc(torch.argmax(cls_score, dim=1), self.num_cluster).cpu()
        if self.label_smoothing > 0 or pred_based_hist is not None:
            loss = ce_smooth(cls_score, target, self.label_smoothing, pred_based_hist)
        else:
            loss = self.loss_fn_cls(cls_score, target) * self.loss_weight["loss_cls"]

        if self.entropy:
            prob_mean = cls_score.mean(dim=0)
            prob_mean[(prob_mean < self.EPS).data] = self.EPS
            loss_ent = (prob_mean * torch.log(prob_mean)).sum()
            loss = loss + loss_ent * self.loss_weight["loss_ent"]

        return loss, curr_hist
