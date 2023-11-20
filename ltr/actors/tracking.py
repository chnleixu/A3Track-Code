from . import BaseActor
import torch
import numpy as np

class TranstActor(BaseActor):
    """ Actor for training the TransT"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        import time
        outputs = self.net(data['search_images'], data['template_images'])

        # generate labels
        targets =[]
        targets_origin = data['search_anno']
        for i in range(len(targets_origin)):
            h, w =data['search_images'][i][0].shape
            target_origin = targets_origin[i]
            target = {}
            target_origin = target_origin.reshape([1,-1])
            target_origin[0][0] += target_origin[0][2] / 2
            target_origin[0][0] /= w
            target_origin[0][1] += target_origin[0][3] / 2
            target_origin[0][1] /= h
            target_origin[0][2] /= w
            target_origin[0][3] /= h
            target['boxes'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['labels'] = label
            targets.append(target)

        # Compute loss
        # outputs:(center_x, center_y, width, height)
        loss_dict = self.objective(outputs, targets)
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'iou': loss_dict['iou'].item()
                 }

        return losses, stats

class DeepActor(BaseActor):
    """ Actor for training the TransT"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        import time
        outputs = self.net(data['search_images'], data['template_images'])

        # generate labels
        targets =[]
        targets_origin = data['search_anno']
        for i in range(len(targets_origin)):
            h, w =data['search_images'].shape[-2:]
            target_origin = targets_origin[i]
            target = {}
            target_origin = target_origin.reshape([1,-1])
            target_origin[0][0] += target_origin[0][2] / 2
            target_origin[0][0] /= w
            target_origin[0][1] += target_origin[0][3] / 2
            target_origin[0][1] /= h
            target_origin[0][2] /= w
            target_origin[0][3] /= h
            target['boxes'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['labels'] = label
            targets.append(target)

        # Compute loss
        # outputs:(center_x, center_y, width, height)
        loss_dict = self.objective(outputs, targets)
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'iou': loss_dict['iou'].item()
                 }

        return losses, stats

class DeepRedActor(BaseActor):
    """ Actor for training the TransT"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        import time
        outputs = self.net(data['search_images'], data['template_images'])

        # generate labels
        targets =[]
        targets_origin = data['search_anno']
        for i in range(len(targets_origin)):
            h, w =data['search_images'].shape[-2:]
            target_origin = targets_origin[i]
            target = {}
            target_origin = target_origin.reshape([1,-1])
            target_origin[0][0] += target_origin[0][2] / 2
            target_origin[0][0] /= w
            target_origin[0][1] += target_origin[0][3] / 2
            target_origin[0][1] /= h
            target_origin[0][2] /= w
            target_origin[0][3] /= h
            target['boxes'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['labels'] = label
            targets.append(target)

        # Compute loss
        # outputs:(center_x, center_y, width, height)
        losses = []
        loss_dict = self.objective(outputs, targets)
        weight_dict = self.objective.weight_dict
        for i in range(len(loss_dict)):
            loss = sum(loss_dict[i][k] * weight_dict[k] for k in loss_dict[i].keys() if k in weight_dict)
            losses.append(loss)
        
        losses = sum(losses)/len(losses)

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce_0': loss_dict[0]['loss_ce'].item(),
                 'Loss/ce_-1': loss_dict[-1]['loss_ce'].item(),
                 'Loss/delta_0': loss_dict[0]['loss_delta'].item(),
                 'Loss/delta_-1': loss_dict[-1]['loss_delta'].item(),
                 'iou_0': loss_dict[0]['iou'].item(),
                 'iou_-1': loss_dict[-1]['iou'].item()
                 }

        return losses, stats

class DeepRedActor(BaseActor):
    """ Actor for training the TransT"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        import time
        outputs = self.net(data['search_images'], data['template_images'])

        # generate labels
        targets =[]
        targets_origin = data['search_anno']
        for i in range(len(targets_origin)):
            h, w =data['search_images'].shape[-2:]
            target_origin = targets_origin[i]
            target = {}
            target_origin = target_origin.reshape([1,-1])
            target_origin[0][0] += target_origin[0][2] / 2
            target_origin[0][0] /= w
            target_origin[0][1] += target_origin[0][3] / 2
            target_origin[0][1] /= h
            target_origin[0][2] /= w
            target_origin[0][3] /= h
            target['boxes'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['labels'] = label
            targets.append(target)

        # Compute loss
        # outputs:(center_x, center_y, width, height)
        losses = []
        loss_dict = self.objective(outputs, targets)
        weight_dict = self.objective.weight_dict
        for i in range(len(loss_dict)):
            loss = sum(loss_dict[i][k] * weight_dict[k] for k in loss_dict[i].keys() if k in weight_dict)
            losses.append(loss)
        
        # total_loss = sum(losses)/len(losses)
        
        total_loss = 0.0
        gamma = 0.5
        n_loss = len(losses)
        for i in range(n_loss):
            loss_weight = gamma**(n_loss - 1 - i)
            total_loss += loss_weight * losses[i]


        # Return training stats
        stats = {'Loss/total': total_loss.item(),
                 'Loss/ce_0': loss_dict[0]['loss_ce'].item(),
                 'Loss/ce_-1': loss_dict[-1]['loss_ce'].item(),
                 'iou_0': loss_dict[0]['iou'].item(),
                 'iou_-1': loss_dict[-1]['iou'].item()
                 }

        return total_loss, stats

