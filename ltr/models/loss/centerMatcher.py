# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import math
import torch
from torch import nn
import numpy as np

class TrackingMatcher(nn.Module):
    """This class computes an assignment between the ground-truth and the predictions of the network.
    The corresponding feature vectors within the ground-truth box are matched as positive samples.
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        
        Params:
            outputs: This is a dict that contains these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Always tensor([0]) represents the foreground,
                           since single target tracking has only one foreground category
                 "boxes": Tensor of dim [1, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order),
                  and it is always 0, because single target tracking has only one target per image
            For each batch element, it holds:
                len(index_i) = len(index_j)
        """
        indices = []
        centernesses = []
        bs, num_queries = outputs["bilinear_coords"].shape[:2]
        bilinear_coords = outputs["bilinear_coords"]
        delta =np.zeros((bs, num_queries, 2), dtype=float)
        for i in range(bs):
            cx, cy, w, h = targets[i]['boxes'][0]
            cx = cx.item(); cy = cy.item(); w = w.item(); h = h.item()
            xmin = cx-w/2; ymin = cy-h/2; xmax = cx+w/2; ymax = cy+h/2

            coords = bilinear_coords[i]

            left = coords[:,0] - xmin
            right = xmax - coords[:,0]
            top = coords[:, 1] - ymin
            down = ymax - coords[:, 1]
            
            index_x = (coords[:,0] - xmin) * (coords[:,0] - xmax)
            index_y = (coords[:,1] - ymin) * (coords[:,1] - ymax)
            index = (left > 0) & (right > 0)&(top>0)&(down>0)
            index2 = (index_x<0)&(index_y<0)

            centerness_x = ((left[index]+right[index])/2 - abs(left[index]-right[index])/2)/((left[index]+right[index])/2 + abs(left[index]-right[index])/2)
            centerness_y = ((top[index]+down[index])/2 - abs(top[index]-down[index])/2)/((top[index]+down[index])/2 + abs(top[index]-down[index])/2)
            centerness = np.sqrt(centerness_x*centerness_y)
            centernesses.append(centerness)
            

            a = np.arange(0, num_queries, 1)
            c = a[index]
            d = np.zeros(len(c), dtype=int)
            indice = (c,d)
            
            

            # _c = function(targets, i)
            # d = np.zeros(len(c), dtype=int)
            # indice = (c,d)
            indices.append(indice)
  
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], torch.cat([torch.as_tensor(centerness, dtype=torch.float64) for centerness in centernesses])

def function(targets, i):
    cx, cy, w, h = targets[i]['boxes'][0]
    cx = cx.item(); cy = cy.item(); w = w.item(); h = h.item()
    xmin = cx-w/2; ymin = cy-h/2; xmax = cx+w/2; ymax = cy+h/2
    len_feature = int(np.sqrt(1024))
    Xmin = int(np.ceil(xmin*len_feature))
    Ymin = int(np.ceil(ymin*len_feature))
    Xmax = int(np.ceil(xmax*len_feature))
    Ymax = int(np.ceil(ymax*len_feature))
    if Xmin == Xmax:
        Xmax = Xmax+1
    if Ymin == Ymax:
        Ymax = Ymax+1
    a = np.arange(0, 1024, 1)
    b = a.reshape([len_feature, len_feature])
    c = b[Ymin:Ymax,Xmin:Xmax].flatten()
    return c

def build_matcher():
    return TrackingMatcher()
