import os
import sys
sys.path.append('/data/xiaojin/huawei/sources/zhangjing/FBX20190_FBXFILESDK_LINUX/lib/Python27_ucs4_x64')
from fbx import *
import numpy as np

import openmesh as om
from FBXClass import FBXClass
import FbxCommon

fbxclass = FBXClass('source_test.fbx')

# demo
# target extra a obj from bash

fbxscene = fbxclass.scene
stackcount = fbxscene.GetSrcObjectCount() # 226
stack = fbxscene.GetSrcObject(113) # FbxAnimCurve
ClusterStackList = [] 
for i in range(stackcount):
    stack = fbxscene.GetSrcObject(i)
    print(i,":",stack,stack.GetName())
    if isinstance(stack,FbxCluster):
        ClusterStackList.append(stack)

for cluster in ClusterStackList:
    node = cluster.GetLink()
    weight = cluster.GetControlPointWeights()
    #& index
    print(cluster.GetLinkMode(),node.GetName(),len(weight))
"""
# stack 
stack = AnimStackList[0]
membercount = stack.GetMemberCount()
MemberList = []
for i in range(membercount):
    member = stack.GetMember(i)
    if not isinstance(member,FbxAnimLayer):
        continue
    MemberList.append(member)
# have only one member
member = MemberList[0]
    
curve = fbxscene.GetSrcObject(112)
keycount = curve.KeyGetCount()
for i in range(keycount):
    keyvalue = curve.KeyGetValue(i)
    keytime = curve.KeyGetTime(i).GetTimeString()
    print(i,keyvalue,keytime)

import pdb
pdb.set_trace()
"""
fbxscene = fbxclass.scene
fbxeval =  fbxscene.GetAnimationEvaluator()
count = fbxscene.GetNodeCount()
time = 1
def evaltime(time,count,fbxscene,fbxeval):
    ftime = FbxTime()
    ftime.SetTime(0,0,time)
    points = []
    for i in range(count):
        node = fbxscene.GetNode(i)
        loc = fbxeval.GetNodeGlobalTransform(node,ftime).GetT()
        loc = [loc[0], loc[1], loc[2]]
        points.append(loc)
    skeleton = om.PolyMesh(points=points)
    om.write_mesh("animation"+str(time)+".obj",skeleton)

# for time in range(5):
#     evaltime(time,count,fbxscene,fbxeval)

import pdb
pdb.set_trace()

