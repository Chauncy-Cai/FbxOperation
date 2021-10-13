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

"""
we can get 
    newmesh,[weightMatrix,kingtree_table,clusterdic]
"""

JointsRotation = {}

def Matrix2Numpy(smat):
    tmat = []
    for i in range(4):
        line = []
        for j in range(4):
            line.append(smat[i][j])
        tmat.append(line)
    return np.array(tmat)

def Vec2Numpy(svec):
    tvec = []
    for i in range(4):
        tvec.append(svec[i])
    return np.array(tvec)

def showMatrix(mat):
    print("showMatrix")
    for i in range(4):
        rowval = []
        for j in range(4):
            rowval.append(mat.Get(i,j))
        print(rowval)

def GetGeometry(node):
    T = node.GetGeometricTranslation(node.eSourcePivot)
    R = node.GetGeometricRotation(node.eSourcePivot)
    S = node.GetGeometricScaling(node.eSourcePivot)
    return FbxAMatrix(T,R,S)

def GetPoseMatrix(pPose,pNodeIndex):
    PI = 3.141592653
    Matrix = pPose.GetMatrix(pNodeIndex)
    # copy Matrix to AMatrix
    T = FbxVector4(Matrix[3][0],Matrix[3][1],Matrix[3][2]) 
    thetaz = math.atan2(Matrix[0][1], Matrix[0][0]) / PI * 180
    thetay  =math.atan2(-1 * Matrix[0][2], math.sqrt(Matrix[1][2]**2 + Matrix[2][2]**2)) / PI *180
    thetax = math.atan2(Matrix[1][2], Matrix[2][2]) / PI * 180
    R = FbxVector4(thetax,thetay,thetaz)
    PoseMatrix = FbxAMatrix() #copy the matrix
    PoseMatrix.SetR(R)
    PoseMatrix.SetT(T)
    return PoseMatrix

def ExtractNodeRotation(Node,pTime=None):
    def getR(Matrix):
        PI = 3.141592653
        thetaz = math.atan2(Matrix[0][1], Matrix[0][0]) / PI * 180
        thetay  =math.atan2(-1 * Matrix[0][2], math.sqrt(Matrix[1][2]**2 + Matrix[2][2]**2)) / PI *180
        thetax = math.atan2(Matrix[1][2], Matrix[2][2]) / PI * 180
        return [thetax,thetay,thetaz]
    global JointsRotation 
    if pTime is not None:
        pTime0 = FbxTime()
        pTime0.SetTime(0,0,0)
        pNode = Node.GetParent()
        OriLocalPosition = FbxAMatrix(Node.EvaluateLocalTransform(pTime0))
        OriGlobalPosition = FbxAMatrix(Node.EvaluateGlobalTransform(pTime0))
        LocalPosition = FbxAMatrix(Node.EvaluateLocalTransform(pTime))
        GlobalPosition = FbxAMatrix(Node.EvaluateGlobalTransform(pTime))
        nodename = Node.GetName()
        if pNode is not None:
            OriParentPosition =  FbxAMatrix(pNode.EvaluateGlobalTransform(pTime0)) 
        if pNode:#sum(abs(theta-theta0))>0:
            Rotation = OriParentPosition * OriLocalPosition * LocalPosition.Inverse() * OriParentPosition.Inverse()
            theta = np.array(getR(Rotation.Inverse()))
            JointsRotation[nodename] = theta
        else:
            Rotation = GlobalPosition
            theta = np.array(getR(Rotation.Inverse())) 
            JointsRotation[nodename] = theta
    else:
        print("not finished yet")
        return 

def GetGlobalPosition(Node,pTime,pPose,pParentGloablPosition=None):
    pPose=None
    if pPose:
        print("not finish yet")
    else:
        GlobalPosition = Node.EvaluateGlobalTransform(pTime)
        ExtractNodeRotation(Node,pTime)
    return  GlobalPosition

def ComputeClusterDeformation(GlobalPosition,Mesh,Cluster,pTime,pPose): 
    #check
    ReferenceGlobalInitPosition = FbxAMatrix()
    ReferenceGlobalCurrentPosition = FbxAMatrix()
    ClusterGlobalInitPosition = FbxAMatrix()
    ClusterGlobalInitPosition = FbxAMatrix()
    
    ReferenceGeometry = FbxAMatrix()
    ClusterGeometry = FbxAMatrix()
    ClusterRelativeInitPosition = FbxAMatrix()
    ClusterRelativeCurrentPositionInverse = FbxAMatrix()
    
    Cluster.GetTransformMatrix(ReferenceGlobalInitPosition)
    ReferenceGlobalCurrentPosition = GlobalPosition
    ReferenceGeometry = GetGeometry(Mesh.GetNode())
    ReferenceGlobalInitPosition *= ReferenceGeometry

    Cluster.GetTransformLinkMatrix(ClusterGlobalInitPosition)
    ClusterGlobalCurrentPosition = GetGlobalPosition(Cluster.GetLink(),pTime,pPose)

    ClusterRelativeInitPosition = ClusterGlobalInitPosition.Inverse() * ReferenceGlobalInitPosition
    ClusterRelativeCurrentPositionInverse = ReferenceGlobalCurrentPosition.Inverse() * ClusterGlobalCurrentPosition
    VertexTransformMatrix = ClusterRelativeCurrentPositionInverse * ClusterRelativeInitPosition

    return VertexTransformMatrix

def ExtractPolygon(mesh):
    # our model is polygon instead of triangle
    PolygonCount = mesh.GetPolygonCount()
    polygons = []
    for i in range(PolygonCount):
        vertexs = []
        for j in range(4):
            v = mesh.GetPolygonVertex(i,j)
            vertexs.append(v)
        polygons.append(vertexs)
    return polygons


def ExtractObjFromFbx(fbxfile,time=None):
    # fbxfile: file name of fbx
    # time: count in second [second,frame]
    # TARGET: extract mesh from fbx with time
    fbxclass = FBXClass(fbxfile)
    fbxscene = fbxclass.scene
    fbxeval = fbxscene.GetAnimationEvaluator()

    mesh =  fbxscene.GetGeometry(0) #get mesh
    if mesh is None:
        print("error")
        import pdb
        pdb.set_trace()
    srcVertex = mesh.GetControlPoints() # original vertex
    pTime = FbxTime()
    pTime.SetTime(0,0,time[0],time[1])
    posT = FbxVector4(0,0,0)
    pPose = fbxscene.GetPose(0)
    node = fbxscene.GetNode(0)
    GlobalPosition = GetGlobalPosition(node,pTime,pPose)
    
    index2matrix = {}
    index2weights= {}
    vertexlist = []
    ori = []
    vertexcount = mesh.GetControlPointsCount()
    
    validcount = 0
    invalidcount = 0
    deformer = mesh.GetDeformer(0)
    clustercount = deformer.GetClusterCount()
    weightMatrix = np.zeros((vertexcount,clustercount+1)) # extra one for root
    kingtree_table = np.ones((clustercount+1))*-1
    
    clusterdic = {}
    tempindex = 0
    for clusterindex in range(clustercount):
        cluster = deformer.GetCluster(clusterindex)
        cls0 = cluster.GetLink()
        clsname = cls0.GetName()
        if clsname in clusterdic:
            continue
        stack = [clsname]
        while cls0.GetParent() is not None:
            cls0 = cls0.GetParent()
            cls0name = cls0.GetName()
            if cls0name in clusterdic:
                break
            stack.append(cls0name)
        while len(stack)!=0:
            clsname = stack.pop()
            if clsname in clusterdic:
                continue
            clusterdic[clsname] = tempindex
            tempindex += 1
    for clusterindex in range(clustercount):
        cluster = deformer.GetCluster(clusterindex)
        cls0 = cluster.GetLink()
        clsname = cls0.GetName()
        clsindex = clusterdic[clsname]
        clsp = cls0.GetParent()
        if clsp: #not None/ RootNode
            clspname = clsp.GetName()
            clspindex = clusterdic[clspname]
        else:
            clspindex = -1

        kingtree_table[clsindex] = clspindex
    ## main process
    for clusterindex in range(deformer.GetClusterCount()):
        cluster = deformer.GetCluster(clusterindex)
        if (not cluster.GetLink()):
            print("warming")
            continue
        # compute cluster Deformationi
        # to implete
        if cluster.GetLinkMode()!=0:
            print("error")
        clusterDeformed = ComputeClusterDeformation(GlobalPosition,mesh,cluster,pTime,pPose)
        clusterVertexCount = cluster.GetControlPointIndicesCount()
        node = cluster.GetLink()
        nodename = node.GetName()
        cluster2index = clusterdic[nodename]
        for j in range(clusterVertexCount):
            index = cluster.GetControlPointIndices()[j]
            weight = cluster.GetControlPointWeights()[j]
            if weight==0.0:
                continue
            if index>=vertexcount:
                print("warming")
                continue
            # not consider other format!!
            weightMatrix[index][cluster2index] = weight
            if index in index2matrix:
                index2matrix[index] += Matrix2Numpy(clusterDeformed)*weight
            else:
                index2matrix[index] = Matrix2Numpy(clusterDeformed)*weight
            if index in index2weights:
                index2weights[index] += weight
            else:
                index2weights[index] = weight
    for j in range(vertexcount):
        srcvertex = Vec2Numpy(srcVertex[j])
        srcvertex[-1]=1
        ori.append(srcvertex[:-1])
        if j in index2matrix:
            # add T or not
            dst = index2matrix[j].T.dot(srcvertex)
            dst = dst*1.0/index2weights[j]
            validcount += 1            
        else:
            dst = srcvertex
            invalidcount += 1
        vertexlist.append(dst[:-1])
    newmesh = om.PolyMesh(points=vertexlist,face_vertex_indices=ExtractPolygon(mesh))
    return newmesh,[weightMatrix,kingtree_table,clusterdic]



if __name__=='__main__':
    # fbx from haiguai please use (second=0)
    # newmesh, info = ExtractObjFromFbx('source_test.fbx',time=[1,0])
    # newmesh, info = ExtractObjFromFbx('fbxfolder/x1.FBX',time=[0,30])
    # newmesh, info = ExtractObjFromFbx('fbxfolder/new_skinning.fbx',time=[0,0])
    # weightMatrix,kingtree_table,clusterdic =  info
    # np.save("weightMatrix.npy",weightMatrix)
    #np.save("ktree_table.npy",kingtree_table)
    #np.save("clusterdic.npy",clusterdic)
    """
    newmesh, info = ExtractObjFromFbx('fbxfolder/x1.FBX',time=[0,30])
    om.write_mesh("x1_30.obj",newmesh)
    newmesh, info = ExtractObjFromFbx('fbxfolder/x1.FBX',time=[0,0])
    om.write_mesh("x1_0.obj",newmesh)
    newmesh, info = ExtractObjFromFbx('fbxfolder/x2.FBX',time=[0,30])
    om.write_mesh("x2_30.obj",newmesh)
    newmesh, info = ExtractObjFromFbx('fbxfolder/x2.FBX',time=[0,0])
    om.write_mesh("x2_0.obj",newmesh)
    newmesh, info = ExtractObjFromFbx('fbxfolder/x3.FBX',time=[0,30])
    om.write_mesh("x3_30.obj",newmesh)
    newmesh, info = ExtractObjFromFbx('fbxfolder/x3.FBX',time=[0,0])
    om.write_mesh("x3_0.obj",newmesh)
    """
    """
    print(len(JointsRotation))
    for key in JointsRotation.keys():
        val = JointsRotation[key]
        val1 = sum(abs(val))
        if val1>2:
            print(key,val)
    """
    def test_obj_extract():
        fbxfolder = './animation/'
        target = './objsFolder/'
        for sub in os.listdir(fbxfolder):
            for f in os.listdir(os.path.join(fbxfolder,sub)):
                if '.fbx' in f:
                    fbxfile = os.path.join(fbxfolder,sub,f)
                    mesh0,info = ExtractObjFromFbx(fbxfile,time=[0,0])
                    mesh30,info = ExtractObjFromFbx(fbxfile,time=[0,30])
                    break
            name0 = f.split('.')[0] + "_0.obj"
            name30 = f.split('.')[0] + "_30.obj"
            
            om.write_mesh(target+name0,mesh0)
            om.write_mesh(target+name30,mesh30)
            print('finish '+f) 
     
    # test_obj_extract()
    newmesh, info = ExtractObjFromFbx('01_1_t.FBX',time=[0,30])
    om.write_mesh("temp.obj",newmesh)

