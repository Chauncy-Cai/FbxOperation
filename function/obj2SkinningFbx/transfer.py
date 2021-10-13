import os
import sys

import numpy as np
import openmesh as om

sys.path.append('../../lib/FBX20190_FBXFILESDK_LINUX/lib/Python27_ucs4_x64')
sys.path.append('../../lib/')
from fbx import *
from FBXClass import FBXClass
from SkeletonTransfer import SkeletonTransfer
from Transform import Transform

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-g","--generate_new_only",help="will not generate file repeatedly",action="store_true")
parser.add_argument("-r","--root_address",help="root address",type=str, default="./20210719")
parser.add_argument("-s","--source_fbx_file",help="the name of the source_fbx_file",type=str,default='All')
args = parser.parse_args()
print(args.source_fbx_file,args.root_address)
curroot = "/data/xiaojin/huawei/sources/zhangjing"
source_list = []
for filename in os.listdir(curroot):
    if 'fbx' in filename or 'FBX' in filename:
        source_list.append(filename)
        
if args.source_fbx_file=="All":
    source_list = source_list
elif args.source_fbx_file in source_list:
    source_list = [args.source_fbx_file]
else:
    print("error")
    exit(-1)

for animation in source_list:
    tarname = animation.split('_')[-1]

    root = curroot + "/" + args.root_address
    for sub in os.listdir(root):
        data_dir = os.path.join(root, sub)
        for d in os.listdir(data_dir):
            obj_path = os.path.join(data_dir, d, "check", "m.OBJ")
            fbx_path = os.path.join(data_dir, d, "check", tarname)
            if os.path.exists(fbx_path) and args.generate_new_only:
                continue
            print(fbx_path)
            target = om.read_polymesh(obj_path) # target
            transformer = Transform("template_color.obj") #source 
            target_keypoints = transformer.generateSkeleton2Pos(target)
            target_vertices = target.points()
            # I can not find a deep-copy operation in the fbx sdk, so I re-read it in each iteration
            fbx_class = FBXClass(animation)
            s = SkeletonTransfer()
            s.transfer(fbx_class, target_vertices, target_keypoints)
            fbx_class.save(fbx_path, pFileFormat=0)
            if not os.path.exists(fbx_path):
                print("???")


