import torch
import torch.nn.functional as F
from dexi_utils import *


def _weighted_cross_entropy_loss(preds, edges):
    """ Calculate sum of weighted cross entropy loss. """
    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7s

    mask = (edges > 0.5).float()

    b, c, h, w = mask.shape

    num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
    num_neg = c * h * w - num_pos                     # Shape: [b,].

    weight = torch.zeros_like(mask)
    weight[edges > 0.5] = num_neg / (num_pos + num_neg)
    weight[edges <= 0.5] = num_pos / (num_pos + num_neg)

    # Calculate loss
    losses = F.binary_cross_entropy_with_logits(preds.float(),
                                                edges.float(),
                                                weight=weight,
                                                reduction='none')
    # loss = torch.sum(losses) / b
    loss = torch.mean(losses)
    return loss


def weighted_cross_entropy_loss(preds, edges):
    """ Calculate sum of weighted cross entropy loss. """

    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7
    # edges= torch.cat([edge,edge,edge,edge,edge,edge,edge], dim=0)
    # print(preds.shape, edges.shape)
    mask = (edges > 0.1).float()
    b,c, h, w = mask.shape

    # Shape: [b,].
    num_pos = torch.sum(mask, dim=[1, 2,3], keepdim=True).float()
    # print("pos", num_pos.shape)

    num_neg = c*h * w - num_pos                     # Shape: [b,].
    # print("neg", num_neg.shape)
    weight = torch.zeros_like(mask)
    #weight[edges > 0.5]  = num_neg / (num_pos + num_neg)
    #weight[edges <= 0.5] = num_pos / (num_pos + num_neg)
    weight.masked_scatter_(edges > 0.1,
                           torch.ones_like(edges) * num_neg / (num_pos + num_neg))
    weight.masked_scatter_(edges <= 0.1,
                           torch.ones_like(edges) * num_pos / (num_pos + num_neg))

    # Calculate loss
    # preds=torch.sigmoid(preds)
    losses = F.binary_cross_entropy_with_logits(preds.float(),
                                                edges.float(),
                                                weight=weight,
                                                reduction='none')
    losses=losses.sum(dim=[1,2,3],keepdim=True)
    # beta = num_neg / (num_neg + num_pos)
    # cost = torch.mean(losses * (1 - beta))
    cost = torch.mean(losses)
    # print('after', cost)
    loss_weight = 1.0  # torch.tensor([1.0]).repeat(c).cuda()

    loss = cost * loss_weight
    # print("loss>", loss)
    return loss
