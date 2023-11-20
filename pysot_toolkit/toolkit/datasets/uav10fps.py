import json
import os
import numpy as np

from PIL import Image
from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video

def ca():
    path='/home/richard_lei/testUAV/UAV123@10fps'
    
    name_list=os.listdir(path+'/data_seq')
    name_list.sort()
    a=len(name_list)
    b=[]
    for i in range(a):
        b.append(name_list[i])
    c=[]
    
    for jj in range(a):
        imgs=path+'/data_seq/'+str(name_list[jj])
        txt=path+'/anno/'+str(name_list[jj])+'.txt'
        bbox=[]
        f = open(txt)               # 返回一个文件对象
        file= f.readlines()
        li=os.listdir(imgs)
        li.sort()
        for ii in range(len(file)):
            li[ii]=name_list[jj]+'/'+li[ii]
    
            line = file[ii].strip('\n').split(',')
            
            try:
                line[0]=int(line[0])
            except:
                line[0]=float(line[0])
            try:
                line[1]=int(line[1])
            except:
                line[1]=float(line[1])
            try:
                line[2]=int(line[2])
            except:
                line[2]=float(line[2])
            try:
                line[3]=int(line[3])
            except:
                line[3]=float(line[3])
            bbox.append(line)
            
        if len(bbox)!=len(li):
            print (jj)
        f.close()
        c.append({'attr':[],'gt_rect':bbox,'img_names':li,'init_rect':bbox[0],'video_dir':name_list[jj]})
        
    d=dict(zip(b,c))

    return d

class UAVVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False):
        super(UAVVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)


class UAV10Dataset(Dataset):
    """
    Args:
        name: dataset name, should be 'UAV123', 'UAV20L'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(UAV10Dataset, self).__init__(name, dataset_root)
        meta_data = ca()

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = UAVVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          meta_data[video]['attr'])

        # set attr
        attr = []
        for x in self.videos.values():
            attr += x.attr
        attr = set(attr)
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
        for x in attr:
            self.attr[x] = []
        for k, v in self.videos.items():
            for attr_ in v.attr:
                self.attr[attr_].append(k)

class VOTVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        camera_motion: camera motion tag
        illum_change: illum change tag
        motion_change: motion change tag
        size_change: size change
        occlusion: occlusion
    """
    def __init__(self, name, root, video_dir, init_rect, img_names, gt_rect,
            load_img=False):
        super(VOTVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, None, load_img)
        self.tags= {'all': [1] * len(gt_rect)}

        # TODO
        # if len(self.gt_traj[0]) == 4:
        #     self.gt_traj = [[x[0], x[1], x[0], x[1]+x[3]-1,
        #                     x[0]+x[2]-1, x[1]+x[3]-1, x[0]+x[2]-1, x[1]]
        #                         for x in self.gt_traj]

        # empty tag
        all_tag = [v for k, v in self.tags.items() if len(v) > 0 ]
        self.tags['empty'] = np.all(1 - np.array(all_tag), axis=1).astype(np.int32).tolist()
        # self.tags['empty'] = np.all(1 - np.array(list(self.tags.values())),
        #         axis=1).astype(np.int32).tolist()

        self.tag_names = list(self.tags.keys())
        if not load_img:
            img_name = os.path.join(root, self.img_names[0])
            img = np.array(Image.open(img_name), np.uint8)
            self.width = img.shape[1]
            self.height = img.shape[0]

    def select_tag(self, tag, start=0, end=0):
        if tag == 'empty':
            return self.tags[tag]
        return self.tags[tag][start:end]

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            traj_files = glob(os.path.join(path, name, 'baseline', self.name, '*0*.txt'))
            if len(traj_files) == 15:
                traj_files = traj_files
            else:
                traj_files = traj_files[0:1]
            pred_traj = []
            for traj_file in traj_files:
                with open(traj_file, 'r') as f:
                    traj = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
                    pred_traj.append(traj)
            if store:
                self.pred_trajs[name] = pred_traj
            else:
                return pred_traj

class UAV10VOTDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'VOT2018', 'VOT2016', 'VOT2019'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(UAV10VOTDataset, self).__init__(name, dataset_root)
        meta_data = ca()

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = VOTVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          load_img=load_img)

        self.tags = ['all', 'empty']