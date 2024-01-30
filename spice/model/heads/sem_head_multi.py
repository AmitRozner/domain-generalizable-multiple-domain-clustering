import torch
import torch.nn as nn
from ..feature_modules.build_feature_module import build_feature_module

import numpy as np
import torch.nn.functional as F
from spice.model.heads import build_head
from spice.model.heads.sem_head import SemHead

import matplotlib.pyplot as plt


class SemHeadMulti(nn.Module):
    def __init__(self, args, multi_heads, ratio_confident=0.90, num_neighbor=100, score_th=0.99, center_based_truncate=False,center_based_truncate_cos=False,
                 **kwargs):

        super(SemHeadMulti, self).__init__()
        self.args = args
        self.num_heads = len(multi_heads)
        self.num_cluster = multi_heads[0].num_cluster
        self.center_ratio = multi_heads[0].center_ratio
        self.num_neighbor = num_neighbor
        self.ratio_confident = ratio_confident
        self.center_based_truncate = center_based_truncate
        self.center_based_truncate_cos = center_based_truncate_cos
        self.score_th = score_th
        if self.args.pred_based_smoothing:
            self.hist_per_head = torch.zeros((self.num_heads, self.num_cluster)) + 1 / self.num_cluster
        for h in range(self.num_heads):
            if "self_smoothing" not in kwargs:
                head_h = SemHead(**multi_heads[h])
            else:
                head_h = SemHead(**multi_heads[h], label_smoothing=kwargs["self_smoothing"])
            self.__setattr__("head_{}".format(h), head_h)

    def local_consistency(self, feas_sim, scores, domain_labels):
        labels_pred = scores.argmax(dim=1).cpu()
        sim_mtx = torch.einsum('nd,cd->nc', [feas_sim.cpu(), feas_sim.cpu()])
        scores_k, idx_k = sim_mtx.topk(k=self.num_neighbor, dim=1)
        labels_samples = torch.zeros_like(idx_k)
        for s in range(self.num_neighbor):
            labels_samples[:, s] = labels_pred[idx_k[:, s]]

        if domain_labels is not None:
            nn_domain_labels = domain_labels[idx_k]
            idx_domain = (np.sum(nn_domain_labels,axis=1) > 0) * (np.sum(nn_domain_labels,axis=1) < self.num_neighbor)
        else:
            idx_domain = True

        true_mtx = labels_samples[:, 0:1] == labels_samples
        num_true = true_mtx.sum(dim=1)
        idx_true = num_true >= self.num_neighbor * self.ratio_confident
        idx_conf = scores.max(dim=1)[0].cpu() > self.score_th
        idx_true = idx_true * idx_conf * idx_domain
        idx_select = torch.where(idx_true > 0)[0]
        labels_select = labels_pred[idx_select]

        idx_per_cluster, label_per_cluster, num_per_cluster = \
            self.extract_per_cluster(idx_select, labels_select, np.ones(self.num_cluster))
        curr_num_neighbors = self.num_neighbor
        ratio_conf = self.ratio_confident
        while np.any(np.array(num_per_cluster) < curr_num_neighbors * ratio_conf) and curr_num_neighbors > 2:
            cluster_to_expand = np.array(num_per_cluster) < self.num_neighbor * ratio_conf
            curr_num_neighbors = curr_num_neighbors - max(1,self.num_neighbor//10)
            ratio_conf = ratio_conf - 0.05
            curr_idx_true = num_true >= curr_num_neighbors * ratio_conf
            idx_conf = scores.max(dim=1)[0].cpu() > ratio_conf
            curr_idx_true = curr_idx_true * idx_conf * idx_domain
            curr_idx_select = torch.where(curr_idx_true > 0)[0]
            curr_labels_select = labels_pred[curr_idx_select]
            temp_idx_per_cluster, temp_label_per_cluster, temp_num_per_cluster = \
                self.extract_per_cluster(curr_idx_select, curr_labels_select, cluster_to_expand)

            for cluster_ind, expand_cluster in enumerate(cluster_to_expand):
                if expand_cluster:
                    num_per_cluster[cluster_ind] = temp_num_per_cluster[0]
                    temp_num_per_cluster = temp_num_per_cluster[1:]
                    label_per_cluster[cluster_ind] = temp_label_per_cluster[0]
                    temp_label_per_cluster = temp_label_per_cluster[1:]
                    idx_per_cluster[cluster_ind] = temp_idx_per_cluster[0]
                    temp_idx_per_cluster = temp_idx_per_cluster[1:]

        idx_per_cluster_select, label_per_cluster_select = self.truncate_classes(idx_per_cluster, label_per_cluster,
                                                                                 num_per_cluster, feas_sim)

        idx_select = torch.cat(idx_per_cluster_select)
        labels_select = torch.cat(label_per_cluster_select)

        return idx_select, labels_select

    def truncate_classes(self, idx_per_cluster, label_per_cluster, num_per_cluster, embeddings):
        idx_per_cluster_select = []
        label_per_cluster_select = []
        min_cluster = np.array(num_per_cluster).min()
        min_cluster = min_cluster if min_cluster > self.num_neighbor * self.ratio_confident else \
            int(self.num_neighbor * self.ratio_confident)

        for c in range(self.num_cluster):
            idx_shuffle = np.arange(0, num_per_cluster[c])
            if len(idx_shuffle) >= min_cluster:
                if self.center_based_truncate or self.center_based_truncate_cos:
                    idx_shuffle = self.center_based_truncate_func(idx_per_cluster[c],embeddings[idx_per_cluster[c]])
                else:
                    np.random.shuffle(idx_shuffle)

                idx_per_cluster_select.append(idx_per_cluster[c][idx_shuffle[0:min_cluster]])
                label_per_cluster_select.append(label_per_cluster[c][idx_shuffle[0:min_cluster]])
            else:
                idx_per_cluster_select.append(idx_per_cluster[c])
                label_per_cluster_select.append(label_per_cluster[c])

        return idx_per_cluster_select, label_per_cluster_select

    def center_based_truncate_func(self, cluster_ind, embeddings):
        center_embedding = embeddings.mean(0)
        if self.center_based_truncate_cos:
            dist_from_center = 1-torch.cosine_similarity(center_embedding.unsqueeze(0), embeddings,dim=1)
        else:
            dist_from_center = ((embeddings - center_embedding)**2).norm(2,dim=1)
        return torch.argsort(dist_from_center)


    def extract_per_cluster(self, idx_select, labels_select, cluster_to_expand):
        num_per_cluster = []
        idx_per_cluster = []
        label_per_cluster = []
        for ind, c in enumerate(range(self.num_cluster)):
            if cluster_to_expand[ind]:
                idx_c = torch.where(labels_select == c)[0]
                idx_per_cluster.append(idx_select[idx_c])
                num_per_cluster.append(len(idx_c))
                label_per_cluster.append(labels_select[idx_c])

        return idx_per_cluster, label_per_cluster, num_per_cluster

    def compute_cluster_proto(self, feas_sim, scores):

        _, idx_max = torch.sort(scores, dim=0, descending=True)
        idx_max = idx_max.cpu()
        num_per_cluster = idx_max.shape[0] // self.num_cluster
        k = int(self.center_ratio * num_per_cluster)
        idx_max = idx_max[0:k, :]
        centers = []
        for c in range(self.num_cluster):
            centers.append(feas_sim[idx_max[:, c], :].mean(axis=0).unsqueeze(dim=0))

        centers = torch.cat(centers, dim=0)
        return centers

    def select_samples(self, feas_sim, scores, i, equalize_domain_sample_select, domain_labels,use_other_heads,targets_list=None,
                       select_samples_max_k=10,select_samples_v2=False,args=None):
        assert len(scores) == self.num_heads
        idx_select = []
        labels_select = []
        idx_max_list = np.array([])
        for h in range(self.num_heads):
            if equalize_domain_sample_select:
                idx_select_h, labels_select_h,idx_max = \
                    self.__getattr__("head_{}".format(h)).select_samples_domain(feas_sim, scores[h], i,idx_max_list,targets_list, domain_labels,select_samples_max_k)
                if h == 0:
                    idx_max_list = np.transpose(np.array(idx_max.cpu().detach()))
                else:
                    idx_max_list = np.concatenate([idx_max_list, np.transpose(np.array(idx_max.cpu().detach()))],
                                                  axis=0)
            elif use_other_heads and False:# and i>20:
                idx_select_h, labels_select_h = \
                    self.__getattr__("head_{}".format(h)).select_samples_domain(feas_sim, scores[h], i, domain_labels)
            elif select_samples_v2:
                idx_select_h, labels_select_h,idx_max = \
                    self.__getattr__("head_{}".format(h)).select_samples_v2(feas_sim, scores[h], i,args=self.args)
            else:
                idx_select_h, labels_select_h,idx_max = \
                    self.__getattr__("head_{}".format(h)).select_samples(feas_sim, scores[h], i,idx_max_list,targets_list,domain_labels,args=self.args)
            if h == 0:
                idx_max_list = np.transpose(np.array(idx_max.cpu().detach()))
            else:
                idx_max_list = np.concatenate([idx_max_list, np.transpose(np.array(idx_max.cpu().detach()))],
                                              axis=0)

            idx_select.append(idx_select_h)
            labels_select.append(labels_select_h)

        return idx_select, labels_select

    def forward(self, fea, **kwargs):
        cls_score = []
        if isinstance(fea, list):
            assert len(fea) == self.num_heads

        for h in range(self.num_heads):
            if isinstance(fea, list):
                cls_socre_h = self.__getattr__("head_{}".format(h)).forward(fea[h])
            else:
                cls_socre_h = self.__getattr__("head_{}".format(h)).forward(fea)

            cls_score.append(cls_socre_h)

        return cls_score

    def loss(self, x, target, **kwargs):
        assert len(x) == self.num_heads
        assert len(target) == self.num_heads

        loss = {}

        for h in range(self.num_heads):
            if self.args.pred_based_smoothing:
                loss_h, hist_h = self.__getattr__("head_{}".format(h)).loss(x[h], target[h], self.hist_per_head[h, :])
            else:
                loss_h, hist_h = self.__getattr__("head_{}".format(h)).loss(x[h], target[h])

            loss['head_{}'.format(h)] = loss_h
            if self.args.pred_based_smoothing:
                self.hist_per_head[h, :] = self.hist_per_head[h, :] * 0.99 + (hist_h / torch.sum(hist_h)) * 0.01

        return loss
