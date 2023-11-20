import torch.nn as nn
from ltr import model_constructor
import ltr.models.backbone as backbones

import numpy as np
import torch
import torch.nn.functional as F
from util import box_ops
from einops import rearrange
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)
from einops import rearrange
from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.loss.centerAssignment import build_matcher
from ltr.models.neck.featurefusion_network import build_featurefusion_network


class Model(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, backbone, num_classes, window_penalty=0.49):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network: torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py
            num_classes: number of object classes, always 1 for single object tracking
        """
        super().__init__()
        hidden_dim = 256
        backbone_channels = 384
        self.window_penalty = window_penalty

   
        self.input_proj = nn.Conv2d(backbone_channels, hidden_dim, kernel_size=1)
        self.MGPR = MGPR(hidden_dim)
        self.RAF = RAF(hidden_dim)
  
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.coords_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.box_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
     
        self.backbone = backbone

        
     
    def forward(self, search, template):
        """ The forward expects a NestedTensor, which consists of:
               - search.tensors: batched images, of shape [batch_size x T_times x 3 x H_search x W_search]
               - search.mask: a binary mask of shape [batch_size x T_times x H_search x W_search], containing 1 on padded pixels
               - template.tensors: batched images, of shape [batch_size x 3 x H_template x W_template]
               - template.mask: a binary mask of shape [batch_size x H_template x W_template], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits for all feature vectors.
                                Shape= [batch_size x num_vectors x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all feature vectors, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image.

        """
        feature_search = self.backbone(search.reshape(-1, *search.shape[-3:]))[2]
        feature_template = self.backbone(template.reshape(-1, *template.shape[-3:]))[2]

        feature_search = self.input_proj(feature_search)
        feature_template = self.input_proj(feature_template)
      
        context_fn = self.MGPR(feature_template, feature_search)
        b,_, h, w = context_fn.shape
        hidden_fn = rearrange(context_fn, 'b c h w -> b (h w) c')

       
        coords = self.initialize_coords(context_fn)
        iter0_coords = rearrange(coords, 'b c h w -> b (h w) c')
        outputs = []


        ##iteration 1
        for i in range(10):
            exec('iter{}_coords_o=iter{}_coords.clone().detach()'.format(i+1, i))
            # bilinear_coords = coords_o.cpu().numpy()
            exec('iter{}_context = bilinear_sampler(context_fn, iter{}_coords_o.reshape(b, h, w, 2))'.format(i+1, i+1))
            # exec('iter{}_net = self.gru(net, iter{}_context)'.format(i+1,i+1))

            exec('iter{}_context = rearrange(iter{}_context, \'b c h w -> b (h w) c\')'.format(i+1, i+1))
            exec('hidden_fn, iter{}_hs = self.RAF(hidden_fn, iter{}_context)'.format(i+1,i+1))
            
            exec('iter{}_classes = self.class_embed(iter{}_hs)'.format(i+1,i+1))
            exec('iter{}_bboxes = self.box_embed(iter{}_hs).sigmoid()'.format(i+1, i+1))

            
            exec('iter{}_coords = self.coords_embed(iter{}_hs).sigmoid()'.format(i+1, i+1))
            
            exec('iter{}_out = {{\'pred_logits\': iter{}_classes, \'pred_boxes\': iter{}_bboxes, \'pred_coords\': iter{}_coords,\'bilinear_coords\': iter{}_coords_o}}'.format(i+1,i+1,i+1,i+1,i+1))
            exec('outputs.append(iter{}_out)'.format(i+1))

        return outputs
    
    
    def track(self, search):
        feature_search = self.backbone(search.reshape(-1, *search.shape[-3:]))[2]
        feature_search = self.input_proj(feature_search)
        
        context_fn = self.MGPR(self.feature_template, feature_search)
        b,_, h, w = context_fn.shape
        hidden_fn = rearrange(context_fn, 'b c h w -> b (h w) c')
        
        hann_window_h=torch.hann_window(h,periodic=False).reshape(h,1)
        hann_window_w=torch.hann_window(w,periodic=False).reshape(1,w)
        hann_window = hann_window_w * hann_window_h
        hann_window = hann_window.reshape(b,1,*hann_window.shape).to(context_fn.device)
        
     
        coords = self.initialize_coords(context_fn)
        iter0_coords = rearrange(coords, 'b c h w -> b (h w) c')
        outputs = []


        for i in range(10):
            exec('iter{}_coords_o=iter{}_coords.clone().detach()'.format(i+1, i))
            # bilinear_coords = coords_o.cpu().numpy()
            exec('iter{}_context = bilinear_sampler(context_fn, iter{}_coords_o.reshape(b, h, w, 2))'.format(i+1, i+1))
            # exec('iter{}_net = self.gru(net, iter{}_context)'.format(i+1,i+1))

            exec('iter{}_context = rearrange(iter{}_context, \'b c h w -> b (h w) c\')'.format(i+1, i+1))
            exec('hidden_fn, iter{}_hs = self.RAF(hidden_fn, iter{}_context)'.format(i+1,i+1))

            exec('iter{}_window = bilinear_sampler(hann_window, iter{}_coords_o.reshape(b, h, w, 2))'.format(i+1, i+1))

            exec('iter{}_classes = self.class_embed(iter{}_hs)'.format(i+1,i+1))
            exec('iter{}_bboxes = self.box_embed(iter{}_hs).sigmoid()'.format(i+1, i+1))

            
            exec('iter{}_coords = self.coords_embed(iter{}_hs).sigmoid()'.format(i+1, i+1))
            
            exec('iter{}_out = {{\'pred_logits\': iter{}_classes, \'pred_boxes\': iter{}_bboxes, \'window\': iter{}_window,\'pred_coords\': iter{}_coords,\'anchor_coords\': iter{}_coords_o}}'.format(i+1,i+1,i+1,i+1,i+1,i+1))
            exec('outputs.append(iter{}_out)'.format(i+1))
            # exec('scoreMap = self._convert_score(iter{}_classes)'.format(i+1))
            # exec('outputMaps  = {\'scoreMap\': scoreMap}')
        
     
        scoreMap = self._convert_score(outputs[0]['pred_logits'])
        outputMaps  = {'scoreMap': scoreMap}
        return outputs, outputMaps



    
   
    def template(self, template):
        feature_template = self.backbone(template.reshape(-1, *template.shape[-3:]))[2]
        self.feature_template = self.input_proj(feature_template)
    
    
    def correlationFunction(self, templateMap, searchMap):
        b, c, t_h, t_w = templateMap.shape 
        b, c, s_h, s_w = searchMap.shape
        fmap_t = rearrange(templateMap, 'b c t_h t_w -> b (t_h t_w) c')
        fmap_s = rearrange(searchMap, 'b c s_h s_w -> b c (s_h s_w)')

        corr = torch.bmm(fmap_t, fmap_s)
        correlationResult = corr.reshape(b,-1, s_h, s_w)
        return correlationResult
    
    def _convert_score(self, score):
        score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 0] 
        return score.reshape(32, 32)

    
    def initialize_coords(self, featuremap):
        N, C, H, W = featuremap.shape
        coords= self.coords_grid(N, H, W, device=featuremap.device)
        # optical flow computed as difference: flow = coords1 - coords0
        return coords
    
    def coords_grid(self, batch, ht, wd, device):
        coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
        coords = coords[0]/(ht-1.), coords[1]/(wd-1.)
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch, 1, 1, 1)

    
    def coords_process(self, coords):
        coords1, coords2 = torch.split(coords,[2,2], dim=1)
        coords_top_left = (coords1+coords2)/2 - (coords1 - coords2).abs()/2
        coords_bottom_right = (coords1+coords2)/2 + (coords1 - coords2).abs()/2
        coords = torch.cat([coords_top_left, coords_bottom_right], dim=1)
        return coords

    def logits(self, x):
        output = torch.log(x/(1-x))
        return output
    
    def xyxy2cxcywh(self, coords):
        coords1, coords2 = torch.split(coords, [2,2], dim=1)
        cxcy = (coords1 + coords2)/2
        wh = coords2 - coords1
        output = torch.cat([cxcy, wh], dim=1)
        return output

    def cxcywh_to_xyxy(self, bboxes):
        coords1 = bboxes[:, :2, ...] - 0.5 * bboxes[:, 2:, ...]
        coords2 = bboxes[:, :2, ...] + 0.5 * bboxes[:, 2:, ...]
        coords = torch.cat([coords1, coords2], dim=1)
        return coords


    
  
class NeckModule(nn.Module):
    def __init__(self, hidden_dim, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=stride, bias=False) 
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        _, _, H, W = x.size()
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        return F.interpolate(out, size=(H,W), mode='bilinear', align_corners=True)

class GranularityModule(nn.Module):
    def __init__(self, hidden_dim, stride):
        super().__init__()
        self.searchConv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=stride, bias=False) 
        self.searchBN = nn.BatchNorm2d(hidden_dim)
        self.templateConv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.templateBN = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        self.context_embedding = ConvNet(16*16, hidden_dim, hidden_dim,3,1)

    def forward(self, feature_template, feature_search):
        _, _, H, W = feature_search.size()
        feature_search = self.relu(self.searchBN(self.searchConv(feature_search)))
        feature_template = self.relu(self.templateBN(self.templateConv(feature_template)))
        similarityMap = self.correlationFunction(feature_template, feature_search)
        context_fn = self.context_embedding(similarityMap)
        
        return F.interpolate(context_fn, size=(H,W), mode='bilinear', align_corners=True)
    
    def correlationFunction(self, templateMap, searchMap):
        b, c, t_h, t_w = templateMap.shape 
        b, c, s_h, s_w = searchMap.shape
        fmap_t = rearrange(templateMap, 'b c t_h t_w -> b (t_h t_w) c')
        fmap_s = rearrange(searchMap, 'b c s_h s_w -> b c (s_h s_w)')

        corr = torch.bmm(fmap_t, fmap_s)
        correlationResult = corr.reshape(b,-1, s_h, s_w)
        return correlationResult


class MGPR(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.Granularity1 = GranularityModule(hidden_dim, 1)
        self.Granularity2 = GranularityModule(hidden_dim, 2)
        self.Granularity3 = GranularityModule(hidden_dim, 4)
       
        self.embedding = ConvNet(3*hidden_dim, hidden_dim, hidden_dim,3,1)
    
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def forward(self, feature_template, feature_search):
        c1 = self.Granularity1(feature_template, feature_search)
        c2 = self.Granularity2(feature_template, feature_search)
        c3 = self.Granularity3(feature_template, feature_search)
   
        c = torch.cat([c1, c2, c3], dim=1)

        return self.embedding(c)


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=512):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class RAF(nn.Module):
    def __init__(self, hidden_dim=256):
        super(RAF, self).__init__()
        self.layerz = nn.Linear(2*hidden_dim, hidden_dim)
        self.layerr = nn.Linear(2*hidden_dim, hidden_dim)
        self.layerq = nn.Linear(2*hidden_dim, hidden_dim)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=-1)

        z = torch.sigmoid(self.layerz(hx))
        r = torch.sigmoid(self.layerr(hx))
        q = torch.tanh(self.layerq(torch.cat([r*h, x], dim=-1)))

        h = (1-z) * h + z * q
        return h, h

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, rate):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(Block(n, k, rate) for n, k in zip([input_dim] + h, h + [hidden_dim]))
        self.output_layer = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.output_layer(x) 
        return x

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, rate, strides=1):
        super().__init__()
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, dilation=rate, padding=rate, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        out = self.block(x)
        return out





def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid - 1
    ygrid = 2*ygrid - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img
    
@model_constructor
def model_network(settings):
    num_classes = 1
    backbone_net = backbones.alexnet(pretrained=True)

    model = Model(
        backbone_net,
        num_classes=num_classes
    )
    device = torch.device(settings.device)
    model.to(device)
    return model

