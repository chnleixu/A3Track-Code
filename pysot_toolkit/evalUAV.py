from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import sys

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from pysot_toolkit.toolkit.datasets import UAVDataset, DTBDataset, V4RDataset, UAV10Dataset, UAVDTDataset
from pysot_toolkit.toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, EAOBenchmark, F1Benchmark
from pysot_toolkit.toolkit.visualization import draw_success_precision
import numpy as np
parser = argparse.ArgumentParser(description='transt evaluation')
parser.add_argument('--tracker_path', '-p', type=str, default='',
                    help='tracker result path')
parser.add_argument('--dataset', '-d', type=str, default='LaSOT',
                    help='dataset name')
parser.add_argument('--num', '-n', default=1, type=int,
                    help='number of thread to eval')
parser.add_argument('--tracker_prefix', '-t', default='',
                    type=str, help='tracker name')
parser.add_argument('--show_video_level', '-s', dest='show_video_level',
                    action='store_true')
parser.add_argument('--vis', dest='vis', action='store_true')
parser.set_defaults(show_video_level=False)
args = parser.parse_args()

def main():
    # dataset_root = '/home/richard_lx/private2/pysot/testing/UAV123' #Absolute path of the dataset
    # dataset_root = '/media/8T/DTB/DTB70'
    # dataset_root = '/media/richard_lei/85fb67d2-0699-420e-9808-a4a0a463b428/LaSOT'
    # dataset_root = '/home/richard_lei/test/V4RFlight112'
    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = glob(os.path.join(args.tracker_path,
                                 args.dataset,
                                 args.tracker_prefix+'*'))
    trackers = [x.split('/')[-1] for x in trackers]

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    # root = dataset_root

    if 'UAVTrack112' in args.dataset:
        root = '/home/richard_lei/test/V4RFlight112'
        dataset = V4RDataset(args.dataset, root)
    elif 'UAV112L' in args.dataset:
        root = '/home/richard_lei/test/UAV112L'
        dataset = V4RDataset(args.dataset, root)
    elif 'DTB' in args.dataset:
        # root = '/media/8T/DTB/DTB70'
        root = '/home/richard_lei/testUAV/DTB70'
        dataset = DTBDataset(args.dataset, root)
    elif 'UAV123' in args.dataset:
        root = '/home/richard_lx/private2/pysot/testing/UAV123'
        dataset = UAVDataset(args.dataset, root)
    elif 'UAV10' in args.dataset:
        root = '/home/richard_lei/testUAV/UAV123@10fps/data_seq'
        dataset = UAV10Dataset(args.dataset, root)
    elif 'UAVDT' in args.dataset:
        root = '/home/richard_lei/testUAV/UAVDT'
        dataset = UAVDTDataset(args.dataset, root)
        
        
   
   
    dataset.set_tracker(tracker_dir, trackers)
    benchmark = OPEBenchmark(dataset)
    success_ret = {} 
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
            trackers), desc='eval success', total=len(trackers), ncols=18):
            success_ret.update(ret)
    precision_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
            trackers), desc='eval precision', total=len(trackers), ncols=18):
            precision_ret.update(ret)
    norm_precision_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
            trackers), desc='eval norm precision', total=len(trackers), ncols=18):
            norm_precision_ret.update(ret)
    benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
            show_video_level=args.show_video_level)
    if args.vis:
        for attr, videos in dataset.attr.items():
            if 'UAV10' in dataset.name:
                dataset_name = 'UAV123@10fps'
            elif 'DTB' in dataset.name:
                dataset_name = 'DTB70'
            elif 'UAV112L' in dataset.name:
                dataset_name = 'UAVTrack112\_L'
            else:
                dataset_name = dataset.name
            draw_success_precision(success_ret,
                        name=dataset_name,
                        videos=videos,
                        attr=attr,
                        precision_ret=precision_ret,
                        norm_precision_ret=norm_precision_ret)


if __name__ == '__main__':
    main()