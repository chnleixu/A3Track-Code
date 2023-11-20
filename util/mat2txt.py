import scipy.io as scio
# import pandas as pd
import numpy as np
import os
import shutil
# import h5py
import hdf5storage

# Using the environment of tongji
# ------------------------------
# 1. .mat文件转换成.txt文件
# ------------------------------
# 注意：下面的代码根据不同类型的.mat有所不同  可输出.mat中的keys查看
# path_matfile = "/home/richard_lei/project/ARCF"
# outpath = "/home/richard_lei/project/ARCF_txt"
path_matfile = "/home/richard_lei/project/AutoTrack"
outpath = "/home/richard_lei/project/AutoTrack_txt"
all_matfile_list = os.listdir(path_matfile)

each_matfile_path = [os.path.join(path_matfile, matfile) for matfile in all_matfile_list]
videos_name_txt = [video_name.replace('_AutoTrack', '').replace('.mat', '') + '.txt' for video_name in all_matfile_list]

for i, matfile_path in enumerate(each_matfile_path):
    # data = scio.loadmat(matfile_path)  # 有些类型的mat文件不能读取
    data = hdf5storage.loadmat(matfile_path)  # 兼容mat文件类型较多
    # 下面两行有些.mat不同
    track_result = data['results']  # 有些tracker的键值为'result'
    track_result = track_result[0][0][0][0][1]  # 有些为track_result[0][0][0][1]
    print(videos_name_txt[i])
    np.savetxt(os.path.join(outpath, videos_name_txt[i]), track_result, delimiter=',', fmt='%d')

# # -----------------------
# # 2. 文件重命名
# # -----------------------
# # 有些trackers在OTB100或OTB50中生成的是小写字母开头的跟踪结果，需要把首字母转化成大写字母，以匹配otb100.json或otb50.json（因为我下载的.json文件视频的首字母都是大写的）
# # 代码依实际情况变化
# src_path = "E:\\0_OTB_benchmark\\Pysot\\pysot-toolkit\\result\\Results_OTB100\\DaSiamRPN"
# dst_path = "E:\\0_OTB_benchmark\\Pysot\\pysot-toolkit\\result\\Results_OTB100\\DaSiamRPN_upper"
# all_filename = os.listdir(src_path)

# all_filename_path_old = [os.path.join(src_path, filename) for filename in all_filename if '.txt' in filename]
# all_filename_path_new = [os.path.join(dst_path, filename.capitalize()) for filename in all_filename]  # .capitalize()将字符串首字母变为大写

# for (old_file, new_file) in zip(all_filename_path_old, all_filename_path_new):
#     os.renames(old_file, new_file)  # 好像把原来的文件夹覆盖掉了，先拷贝一份，以防万一

# # 从OTB100中转移OTB50数据
# src_otb100_path = "E:\\0_OTB_benchmark\\Pysot\\pysot-toolkit\\result\\Results_OTB100"
# dst_otb50_path = "E:\\0_OTB_benchmark\\Pysot\\pysot-toolkit\\result\\Results_OTB50"

# template_tracker = os.path.join(dst_otb50_path, 'Initial_Method')
# template_videos = os.listdir(template_tracker)

# all_trackername = os.listdir(src_otb100_path)
# src_trackers = [os.path.join(src_otb100_path, src_tracker) for src_tracker in all_trackername]

# for tracker_file in all_trackername:
#     if tracker_file not in os.listdir(dst_otb50_path):
#         os.mkdir(os.path.join(dst_otb50_path, tracker_file))

# print(template_videos)
# print(all_trackername)

# for i, tracker in enumerate(src_trackers):
#     each_tracker_videos = os.listdir(tracker)
#     for video in template_videos:
#         shutil.copyfile(os.path.join(tracker, video), os.path.join(dst_otb50_path, all_trackername[i], video))
