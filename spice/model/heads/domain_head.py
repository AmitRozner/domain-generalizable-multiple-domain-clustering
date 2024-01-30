import torch
import torch.nn as nn
from torch.autograd import Function

class DomainClassifierReverse(nn.Module):
    def __init__(self, domain_size_layers,feature_dim, loss_weight, gpu, num_domains=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.loss_weight = loss_weight
        self.loss_fn = torch.nn.NLLLoss().cuda(gpu)
        self.domain_classifier = nn.Sequential()
        for idx,layer in  enumerate(domain_size_layers):
            if idx != 0:
                self.domain_classifier.add_module('do_'+str(idx),(torch.nn.Dropout(p=0.25)))
            self.domain_classifier.add_module('d_fc'+str(idx), nn.Linear(feature_dim, layer))
            self.domain_classifier.add_module('d_bn'+str(idx), nn.BatchNorm1d(layer))
            self.domain_classifier.add_module('d_relu'+str(idx), nn.ReLU(True))
            feature_dim=layer
        self.domain_classifier.add_module('d_fc_last',torch.nn.Linear(feature_dim, num_domains))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax())
        self.domain_classifier.cuda(gpu)

    def forward(self, features, alpha):
        features = features.view(-1, self.feature_dim)
        reverse_feature = ReverseLayerF.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_feature)

        return domain_output

    def loss(self, features, domain_labels, alpha):
        domain_output = self.forward(features, alpha)
        loss = self.loss_fn(domain_output, domain_labels) * self.loss_weight

        return loss

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None