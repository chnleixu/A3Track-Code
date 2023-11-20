from __future__ import absolute_import
from http.client import responses

import numpy as np
import math
import torchvision.transforms.functional as tvisf
import cv2
import torch
import torch.nn.functional as F
import time


class Tracker(object):

    def __init__(self, name, net, window_penalty=0.49, exemplar_size=128, instance_size=256):
        self.name = name
        self.net = net
        self.window_penalty = window_penalty
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size
        self.net = self.net
        self.count = 1
    
    def _convert_heatmap(self, score):
        score = score.permute(2, 1, 0).contiguous().view(-1)
        score = score.data.cpu().numpy()
        return score

    def _convert_score(self, score):

        score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 0].cpu().numpy()
        return score

    def _convert_bbox(self, delta):

        delta = delta.permute(2, 1, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        return delta
    
    def _convert_window(self, window):
         window = window.view(-1)
         window = window.data.cpu().numpy()
         return window


    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: rgb based image
            pos: center position
            model_sz: exemplar size
            original_sz: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        im_patch = im_patch.cuda()
        return im_patch



    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        tic = time.time()
        hanning = np.hanning(32)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        # Initialize
        self.initialize_features()
        bbox = info['init_bbox']
        self.center_pos = np.array([bbox[0] + bbox[2] / 2,
                                    bbox[1] + bbox[3] / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_z = self.size[1] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_z = math.ceil(math.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(image, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(image, self.center_pos,
                                    self.exemplar_size,
                                    s_z, self.channel_average)

        # normalize
        z_crop = z_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.inplace = False
        z_crop[0] = tvisf.normalize(z_crop[0], self.mean, self.std, self.inplace)

        # initialize template feature
        self.net.template(z_crop)

        out = {'time': time.time() - tic}
        
        return out

    def track(self, image, info: dict = None) -> dict:
        # calculate x crop size
        w_x = self.size[0] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_x = self.size[1] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_x = math.ceil(math.sqrt(w_x * h_x))

        # get crop
        x_crop = self.get_subwindow(image, self.center_pos,
                                    self.instance_size,
                                    round(s_x), self.channel_average)

        # normalize
        x_crop = x_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        x_crop[0] = tvisf.normalize(x_crop[0], self.mean, self.std, self.inplace)

        # track
        outputs, outputMaps = self.net.track(x_crop)
        score = self._convert_score(outputs[-1]['pred_logits'])
        pred_bbox = self._convert_bbox(outputs[-1]['pred_boxes'])
        window = self._convert_window(outputs[-1]['window'])

        def change(r):
            
            return np.maximum(r, 1. / (r+1e-5))

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :]* s_x, pred_bbox[3, :]* s_x) /
                     (sz(self.size[0], self.size[1])))

        # aspect ratio penalty
        r_c = change((self.size[0]/(self.size[1]+1e-5)) /
                     (pred_bbox[2, :]/(pred_bbox[3, :]+1e-5)))
        penalty = np.exp(-(r_c * s_c - 1) * 0.04)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.window_penalty) + \
                 window * self.window_penalty

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx]
        bbox = bbox * s_x
        cx = bbox[0] + self.center_pos[0] - s_x / 2
        cy = bbox[1] + self.center_pos[1] - s_x / 2
        width = bbox[2]
        height = bbox[3]

        lr = penalty[best_idx] * score[best_idx] * 0.4
        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr


        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, image.shape[:2])

        # update state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        
        if self.count%20==0 and ('update' in self.name):
            self.update_template(image)
            print(self.count)
        self.count = self.count + 1
        
        

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        out = {'target_bbox': bbox,
               'best_score': pscore}
        return out
    
    def update_template(self, image):
        w_z = self.size[0] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_z = self.size[1] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_z = math.ceil(math.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(image, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(image, self.center_pos,
                                    self.exemplar_size,
                                    s_z, self.channel_average)

        # normalize
        z_crop = z_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.inplace = False
        z_crop[0] = tvisf.normalize(z_crop[0], self.mean, self.std, self.inplace)

        # initialize template feature
        self.net.template(z_crop)

     
    def visible_outputMaps(self, img, outputMaps, bbox):
        ## debug ##
        import os,cv2,numpy
        import matplotlib.pyplot as plt
        def tensor_to_image(image_norm_tensor):
            image_tensor = image_norm_tensor.clone()
            image_tensor[0,:,:] *= 0.229
            image_tensor[1,:,:] *= 0.224
            image_tensor[2,:,:] *= 0.225
            image_tensor[0,:,:] += 0.485
            image_tensor[1,:,:] += 0.456
            image_tensor[2,:,:] += 0.406
            image_tensor *= 255
            image_tensor = image_tensor.permute(1,2,0)
            image_tensor = image_tensor.cpu().data.numpy().astype(numpy.uint8)
            image_tensor = cv2.cvtColor(image_tensor, cv2.COLOR_RGB2BGR)
            return image_tensor

        
        output_dir = os.path.join('/home/richard_lei/Case/debug1/', self.name)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        count = len(os.listdir(output_dir))
        
        fig, ax = plt.subplots(1,len(outputMaps)+1)
        img = tensor_to_image(img.squeeze(0))


        mask_bbox =bbox * img.shape[0]
        x1 = round(mask_bbox[0]-0.5*mask_bbox[2])
        x2 = round(mask_bbox[0]+0.5*mask_bbox[2])
        y1 = round(mask_bbox[1]-0.5*mask_bbox[3])
        y2 = round(mask_bbox[1]+0.5*mask_bbox[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        img = cv2.cvtColor(numpy.array(img), cv2.COLOR_BGR2RGB)  # cv2转PIL
        img = img / 255
        ax[0].axis("off")
        ax[0].imshow(img)
        i=1
        for key, value in outputMaps.items():
            ax[i].axis("off")
            ax[i].title.set_size(1)
            ax[i].imshow(value.squeeze().cpu().data.numpy())
            ax[i].set(title=f'{key}')
            i=i+1
        fig.savefig(os.path.join(output_dir,'{}-search.jpg'.format(count+1)), dpi=300, format='png', transparent=True)
        print('{}-search.jpg'.format(count))

    