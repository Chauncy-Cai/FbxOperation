import os  
import sys

import numpy as np  
import math 
import openmesh as om

sys.path.append('../../lib/FBX20190_FBXFILESDK_LINUX/lib/Python27_ucs4_x64')
from fbx import *
sys.path.append('../../lib/')
from FBXClass import FBXClass  
import FbxCommon

def skinWeightTransfer(source,target):
    sfbxclass = FBXClass(source)
    smesh =  sfbxclass.scene.GetGeometry(0)
    sDeformer = smesh.GetDeformer(0)
    tfbxclass = FBXClass(target)
    tmesh =  tfbxclass.scene.GetGeometry(0)
    tDeformer = tmesh.GetDeformer(0)
    #cluster count should be the same
    clusterCount = sDeformer.GetClusterCount()
    for cindex in range(clusterCount):
        scluster = sDeformer.GetCluster(cindex)
        scls = scluster.GetLink()
        sname = scls.GetName()
        # find corresponding target cluster
        for temp in range(clusterCount):
            tcluster = tDeformer.GetCluster(temp)
            tcls = tcluster.GetLink()
            tname = tcls.GetName()
            if tname==sname:
                print("cluster found")
                break
                
        clusterVertexCount = scluster.GetControlPointIndicesCount()
        #reset control points of tcluster
        tcluster.SetControlPointIWCount(0)
        for j in range(clusterVertexCount):
            sindex = scluster.GetControlPointIndices()[j]
            sweight = scluster.GetControlPointWeights()[j]
            tcluster.AddControlPointIndex(sindex,sweight)
    tfbxclass.save('./'+target[:-4]+'_result.fbx',pFileFormat=0)

#source = "skinfbx_20211010.fbx"
#target = "20210719_design_dog_01_2_target.fbx"
source = "01_s.fbx"
target = "01_1_t.FBX"
skinWeightTransfer(source,target) #store target
