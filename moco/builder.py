# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from spice.model.heads.domain_head import DomainClassifierReverse


class FFN(nn.Module):
    def __init__(self, dim_mlp=512, encoder_q_out_dim=128):
        super(FFN, self).__init__()

        self.fc1 = nn.Linear(dim_mlp, dim_mlp)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim_mlp, encoder_q_out_dim)

    def forward(self, x):
        emb = self.fc1(x)
        out = self.fc2(self.relu(emb))
        return out, emb

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, args, dim=128, K=65536, m=0.999, T=0.07, mlp=False, input_size=96, num_domains=2,
                 debug=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.debug = debug
        self.args = args
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(pretrained=False,num_classes=dim, in_size=input_size)
        self.encoder_k = base_encoder(pretrained=False,num_classes=dim, in_size=input_size)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = FFN()#nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = FFN()#nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.qMult = 1
        if self.args.multi_q:
            self.qMult = num_domains
            self.register_buffer("queue", torch.randn(self.qMult, dim, K))
            self.queue = nn.functional.normalize(self.queue, dim=1)
            self.register_buffer("queue_idx", - torch.ones(self.qMult, K))
        else:
            self.register_buffer("queue", torch.randn(dim, K))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_idx", - torch.ones(K))

        self.Q_lim = [None] * self.qMult
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.use_domain_classifier = args.use_domain_classifier

        if args.use_domain_classifier:
            self.domain_head = DomainClassifierReverse(args.domain_size_layers,args.feature_dim, args.domain_loss_weight, args.gpu,num_domains=num_domains)

        self.distributed = args.distributed

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_BU(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # self.queue[:, ptr:ptr + batch_size] = keys.T
        # print(keys.shape)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(1, 0)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, q_selector=None, sample_idx=None):
        # gather keys before updating queue
        if not self.debug:
            keys = concat_all_gather(keys)
            if q_selector is not None:
                q_selector = concat_all_gather(q_selector)
            if sample_idx is not None:
                sample_idx = concat_all_gather(sample_idx)
        # also for simplicity, here we assume each domain contributes the same number of samples to the batch
        batch_size = keys.shape[0] // self.qMult

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, f'self.K={self.K}, batch_size={batch_size}'  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if q_selector is not None:
            for iQ in range(self.qMult):
                if self.Q_lim[iQ] is not None:
                    local_ptr = ptr % self.Q_lim[iQ]
                else:
                    local_ptr = ptr
                active_keys = keys[q_selector == iQ]

                self.queue[iQ, :, local_ptr:local_ptr + batch_size] = active_keys.T
                self.queue_idx[iQ, local_ptr:local_ptr + batch_size] = sample_idx[q_selector == iQ]
        else:
            if self.Q_lim[0] is not None:
                local_ptr = ptr % self.Q_lim[0]
            else:
                local_ptr = ptr

            self.queue[:, local_ptr:local_ptr + batch_size] = keys.T
            self.queue_idx[local_ptr:local_ptr + batch_size] = sample_idx

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, domain_labels, alpha, q_selector=None, sample_idx=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q, emb_q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            if self.distributed:
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k, _ = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            if self.distributed:
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # negative logits: NxK
        BIG_NUMBER = 10000.0 * self.T  # taking temp into account for a good measure
        l_neg = (torch.zeros((q.shape[0], self.K)).cuda() - BIG_NUMBER)
        if self.args.multi_q:
            for iQ in range(self.qMult):
                if self.Q_lim[iQ] is not None:
                    qlim = self.Q_lim[iQ]
                else:
                    qlim = self.queue[iQ].shape[1]
                ixx = (q_selector == iQ)
                _l_neg = torch.einsum('nc,ck->nk', [q[ixx], self.queue[iQ][:,:qlim].clone().detach()])
                if sample_idx is not None:
                    for ii, indx in enumerate(sample_idx[ixx]):
                        _l_neg[ii, self.queue_idx[iQ][:qlim] == indx] = - BIG_NUMBER
                l_neg[ixx, :qlim] = _l_neg
        else:
            if self.Q_lim[0] is not None:
                qlim = self.Q_lim[0]
            else:
                qlim = self.queue.shape[1]
            _l_neg = torch.einsum('nc,ck->nk', [q, self.queue[:, :qlim].clone().detach()])
            if sample_idx is not None:
                for ii in range(q.shape[0]):
                    _l_neg[ii, self.queue_idx[:qlim] == sample_idx[ii]] = - BIG_NUMBER
            l_neg[:, :qlim] = _l_neg

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        if self.distributed:
            # dequeue and enqueue
            self._dequeue_and_enqueue(k, q_selector if self.args.multi_q else None, sample_idx)
        if self.use_domain_classifier:
            if self.args.dl_lastlayer:
                domain_loss = self.domain_head.loss(q, domain_labels, alpha)
            else:
                domain_loss = self.domain_head.loss(emb_q, domain_labels, alpha)
        else:
            domain_loss = torch.tensor(0)

        return logits, labels, domain_loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
