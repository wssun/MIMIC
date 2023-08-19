import torch
from torch import nn


# implementation reference:https://github.com/GeniusHTX/CLUB
class DV(nn.Module):
    def __init__(self, TNet):
        super(DV, self).__init__()
        self.F_func = TNet

    def forward(self, feature_maps, repres):
        # repres means the representation of the encoder output
        # samples have shape [sample_size, dim], the feature_maps are flattened
        # shuffle and concatenate
        sample_size = repres.shape[0]
        repres_shuffle = repres[torch.randperm(sample_size)]

        T0 = self.F_func(feature_maps, repres)
        T1 = self.F_func(feature_maps, repres_shuffle)

        lower_bound = T0.mean() - torch.log(T1.exp().mean())
        return lower_bound

    def learning_loss(self, feature_maps, repres):
        return -self.forward(feature_maps, repres)
