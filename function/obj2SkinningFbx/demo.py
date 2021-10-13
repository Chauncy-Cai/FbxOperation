import os
import sys

import numpy as np
import openmesh as om 

sys.path.append('../../lib/FBX20190_FBXFILESDK_LINUX/lib/Python27_ucs4_x64')
sys.path.append('../../lib/')
from FBXClass import FBXClass    
from SkeletonTransfer import SkeletonTransfer
from Transform import Transform  

transformer=Transform("template_color1.obj")
"""
target_root = "/home/shengcai/zhangjing_sources/objs/norm/"
for objfile in os.listdir(target_root):
    fbx_class= FBXClass('01.fbx')  
    s = SkeletonTransfer()
    print(objfile)
    filename =  objfile.split('.')[-2]
    objs = om.read_polymesh(target_root+objfile)  # target objs
    target_keypoints = transformer.generateSkeleton2Pos(objs)
    target_v = objs.points() 
    s.transfer(fbx_class,target_v,target_keypoints)  
    fbx_class.save('./result/'+filename+'.fbx',pFileFormat=0)
    del fbx_class
    del s
"""
#target_root = "/home/shengcai/zhangjing_sources/objs/norm/"
fbx_class= FBXClass('skinfbx_20211010.fbx')  # source file input
s = SkeletonTransfer()
objfile = "20210719_design_dog_01.obj"
print(objfile)
filename =  objfile.split('.')[-2]
objs = om.read_polymesh(objfile)  # target objs
target_keypoints = transformer.generateSkeleton2Pos(objs)
target_v = objs.points() 
s.transfer(fbx_class,target_v,target_keypoints)  
fbx_class.save('./'+filename+'_result.fbx',pFileFormat=0)
del fbx_class
del s
