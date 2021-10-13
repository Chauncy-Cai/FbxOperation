from fbx import *
import numpy as np
import openmesh as om
sys.path.append('../../lib/')  
from FBXClass import FBXClass

'''
data_raw = {'mixamorig:LeftLeg': [0.158769934617223, -0.6361428395786383, -0.03738794772835897], 'mixamorig:Neck': [0.0, 0.2934343830073328, -0.06720592839759171], 'mixamorig:Spine1': [0.0, -0.029668360420178813, -0.03928721328175021], 'mixamorig:Spine2': [0.0, 0.12238004210834214, -0.052425436807091816], 'mixamorig:LeftFoot': [0.15786925976164146, -0.8328573056907638, -0.06099840296411638], 'mixamorig:RightShoulder': [-0.06838959455490112, 0.2731862679446191, -0.06716661913436234], 'mixamorig:LeftShoulder': [0.06838959455490112, 0.2731862679446191, -0.06724524511140169], 'mixamorig:Spine': [0.0, -0.16271062195301056, -0.027791272848844528], 'mixamorig:RightToeBase': [-0.13433001814036888, -0.9546316009111477, 0.11354089316209051], 'mixamorig:RightUpLeg': [-0.1469462811946869, -0.340105265378952, -0.01612982153892517], 'mixamorig:LeftForeArm': [0.41171778369897294, 0.20700610575049716, -0.04431365348662009], 'mixamorig:RightLeg': [-0.15876993523114505, -0.636142834271944, -0.036716691062620384], 'mixamorig:RightForeArm': [-0.41171778408019066, 0.20700612458776252, -0.04400120105275057], 'mixamorig:LeftToeBase': [0.13432999655198072, -0.9546307237206888, 0.111734599909937], 'mixamorig:LeftUpLeg': [0.1469462811946869, -0.340105265378952, -0.01579473167657852], 'mixamorig:LeftArm': [0.20306866394252948, 0.23180973420036313, -0.06732386635469897], 'mixamorig:LeftHand': [0.5635990128434321, 0.1819614991341412, 0.012392992847353193], 'mixamorig:RightFoot': [-0.1578692665682868, -0.8328571297521958, -0.06108353024212582], 'mixamorig:LeftToe_End': [0.11989840555153933, -0.9552764577397921, 0.1948817194421206], 'unamed': [0.0, 0.0, 0.0], 'mixamorig:Head': [0.0, 0.4291715041125269, -0.04459494365495026], 'mixamorig:Hips': [0.0, -0.27674686908721924, -0.017937609925866127], 'mixamorig:RightArm': [-0.20306866607883098, 0.23180974115663155, -0.06708799647322473], 'mixamorig:HeadTop_End': [0.0, 0.961149396034524, 0.04402151690680205], 'mixamorig:RightToe_End': [-0.11989843369065009, -0.9552778146234587, 0.19699413012348932], 'mixamorig:RightHand': [-0.5635990208692265, 0.18196152595726, 0.011825720796457731]}
data_target = {'mixamorig:LeftLeg': [0.19146551263092818, -0.6540266315228804, -0.021430224208800204], 'mixamorig:Neck': [0.0, 0.35245993175227125, -0.06565949496487877], 'mixamorig:Spine1': [0.0, -0.03232073178702613, -0.03774766667872774], 'mixamorig:Spine2': [0.0, 0.14875257947165454, -0.05088264670829076], 'mixamorig:LeftFoot': [0.16441401812875045, -0.8232710927100443, -0.05519229176502708], 'mixamorig:RightShoulder': [-0.1027444452047348, 0.3347580079050728, -0.06596835133294364], 'mixamorig:LeftShoulder': [0.1027444452047348, 0.3347580079050728, -0.06535064604739448], 'mixamorig:Spine': [0.0, -0.1907597929239273, -0.0262545607984066], 'mixamorig:RightToeBase': [-0.1723526283150312, -0.9574540672352234, 0.06980648830566419], 'mixamorig:RightUpLeg': [-0.17054346203804016, -0.402091920375824, -0.021858789026737213], 'mixamorig:LeftForeArm': [0.4975248087512829, 0.21623093522038356, -0.015447707586599323], 'mixamorig:RightLeg': [-0.1914655119179155, -0.6540266208859554, -0.022460974737139865], 'mixamorig:RightForeArm': [-0.49752486933583195, 0.21623112930101535, -0.015600770035602174], 'mixamorig:LeftToeBase': [0.1723521112069892, -0.9574535732630359, 0.06710000990297416], 'mixamorig:LeftUpLeg': [0.17054346203804016, -0.402091920375824, -0.019634637981653214], 'mixamorig:LeftArm': [0.308255570237736, 0.29971006140746553, -0.06473293626398335], 'mixamorig:LeftHand': [0.6588356461460728, 0.17191041033208426, 0.028116072259702807], 'mixamorig:RightFoot': [-0.16441377996487191, -0.8232727579656831, -0.05414916005946043], 'mixamorig:LeftToe_End': [0.17294317431393672, -0.9568778358323612, 0.140106626073431], 'mixamorig:Head': [0.0, 0.4651883784743019, -0.041857415704529506], 'mixamorig:Hips': [0.0, -0.3265646994113922, -0.016403328627347946], 'mixamorig:RightArm': [-0.308255587374755, 0.29971016187105426, -0.0665860598038212], 'mixamorig:HeadTop_End': [0.0, 0.9666539672823615, 0.06402471485872957], 'mixamorig:RightToe_End': [-0.17294407875965895, -0.956877578617603, 0.1436928574474825], 'mixamorig:RightHand': [-0.6588357289638186, 0.17191068569591544, 0.028090579230675324]}
def get_translation(name):
    # print(data)
    dx = data_target[name][0] - data_raw[name][0]
    dy = data_target[name][1] - data_raw[name][1]
    dz = data_target[name][2] - data_raw[name][2]
    return dx, dy, dz

def update(name, node):
    dx, dy, dz = get_translation(name)
    # translation matrix
    d1 = FbxAMatrix()
    d1.SetT(FbxVector4(dx, dy, dz))
    tx = node.EvaluateGlobalTransform()
    temp = d1 * tx
    return temp
'''
    
class SkeletonTransfer(object):
    def __init__(self):
        pass

    def CalculateGlobalTransform(self, node):
        lTranlationM = FbxAMatrix()
        lScalingM = FbxAMatrix()
        lScalingPivotM = FbxAMatrix()
        lScalingOffsetM = FbxAMatrix() 
        lRotationOffsetM = FbxAMatrix()
        lRotationPivotM = FbxAMatrix()
        lPreRotationM = FbxAMatrix()
        lRotationM = FbxAMatrix()
        lPostRotationM = FbxAMatrix() 
        lTransform = FbxAMatrix()
        lParentGX = FbxAMatrix()
        lGlobalT = FbxAMatrix()
        lGlobalRS = FbxAMatrix()

        if node == None:
            lTransform.SetIdentity()
            return lTransform

        # Construct translation matrix
        lTranslation = node.LclTranslation.Get()
        lTranlationM.SetT(FbxVector4(lTranslation))

        # Construct rotation matrices
        lRotation = node.LclRotation.Get()
        lPreRotation = node.PreRotation.Get()
        lPostRotation = node.PostRotation.Get()
        lRotationM.SetR(FbxVector4(lRotation))
        lPreRotationM.SetR(FbxVector4(lPreRotation))
        lPostRotationM.SetR(FbxVector4(lPostRotation))

        # Construct scaling matrix
        lScaling = node.LclScaling.Get()
        lScalingM.SetS(FbxVector4(lScaling))

        # Construct offset and pivot matrices
        lScalingOffset = node.ScalingOffset.Get()
        lScalingPivot = node.ScalingPivot.Get()
        lRotationOffset = node.RotationOffset.Get()
        lRotationPivot = node.RotationPivot.Get()
        lScalingOffsetM.SetT(FbxVector4(lScalingOffset))
        lScalingPivotM.SetT(FbxVector4(lScalingPivot))
        lRotationOffsetM.SetT(FbxVector4(lRotationOffset))
        lRotationPivotM.SetT(FbxVector4(lRotationPivot))

        # Calculate the global transform matrix of the parent node
        lParentNode = node.GetParent()
        if(lParentNode):
            lParentGX = self.CalculateGlobalTransform(lParentNode)
        else:
            lParentGX.SetIdentity()

        
        # Construct Global Rotation
        lLRM = FbxAMatrix()
        lParentGRM = FbxAMatrix()
        lParentGR = lParentGX.GetR()
        lParentGRM.SetR(lParentGR)
        lLRM = lPreRotationM * lRotationM * lPostRotationM

        # Construct Global Shear*Scaling
        # FBX SDK does not support shear, to patch this, we use:
        # Shear*Scaling = RotationMatrix.Inverse * TranslationMatrix.Inverse * WholeTranformMatrix
        lLSM = FbxAMatrix()
        lParentGSM = FbxAMatrix() 
        lParentGRSM = FbxAMatrix()
        lParentTM = FbxAMatrix()
        lParentGT = lParentGX.GetT()
        lParentTM.SetT(lParentGT)
        lParentGRSM = lParentTM.Inverse() * lParentGX
        lParentGSM = lParentGRM.Inverse() * lParentGRSM
        lLSM = lScalingM

        # Do not consider translation now
        lInheritType = node.InheritType.Get()
        if lInheritType == FbxTransform.eInheritRrSs:
            lGlobalRS = lParentGRM * lLRM * lParentGSM * lLSM
        elif lInheritType == FbxTransform.eInheritRSrs:
            lGlobalRS = lParentGRM * lParentGSM * lLRM * lLSM
        elif(lInheritType == FbxTransform.eInheritRrs):
            lParentLSM = FbxAMatrix()
            lParentLS = lParentNode.LclScaling.Get()
            lParentLSM.SetS(FbxVector4(lParentLS))
            lParentGSM_noLocal = lParentGSM * lParentLSM.Inverse()
            lGlobalRS = lParentGRM * lLRM * lParentGSM_noLocal * lLSM
        else:
            print("error, unknown inherit type! \n")
        
        # Construct translation matrix
        # Calculate the local transform matrix
        lTransform = lTranlationM * lRotationOffsetM * lRotationPivotM * lPreRotationM * lRotationM * lPostRotationM * lRotationPivotM.Inverse()\
            * lScalingOffsetM * lScalingPivotM * lScalingM * lScalingPivotM.Inverse()
        lLocalTWithAllPivotAndOffsetInfo = lTransform.GetT()
        
        # Calculate global translation vector according to: 
        # GlobalTranslation = ParentGlobalTransform * LocalTranslationWithPivotAndOffsetInfo
        lGlobalTranslation = lParentGX.MultT(lLocalTWithAllPivotAndOffsetInfo)
        lGlobalT.SetT(lGlobalTranslation)

        # Construct the whole global transform
        lTransform = lGlobalT * lGlobalRS

        return lTransform

    def modify(self, node, lParentGX, D=FbxAMatrix()):
        lTranlationM = FbxAMatrix()
        lScalingM = FbxAMatrix()
        lScalingPivotM = FbxAMatrix()
        lScalingOffsetM = FbxAMatrix() 
        lRotationOffsetM = FbxAMatrix()
        lRotationPivotM = FbxAMatrix()
        lPreRotationM = FbxAMatrix()
        lRotationM = FbxAMatrix()
        lPostRotationM = FbxAMatrix() 
        lTransform = FbxAMatrix()
        lGlobalT = FbxAMatrix()
        lGlobalRS = FbxAMatrix()

        if node == None:
            lTransform.SetIdentity()
            return lTransform

        # Construct translation matrix
        lTranslation = node.LclTranslation.Get()
        lTranlationM.SetT(FbxVector4(lTranslation))

        # Construct rotation matrices
        lRotation = node.LclRotation.Get()
        lPreRotation = node.PreRotation.Get()
        lPostRotation = node.PostRotation.Get()
        lRotationM.SetR(FbxVector4(lRotation))
        lPreRotationM.SetR(FbxVector4(lPreRotation))
        lPostRotationM.SetR(FbxVector4(lPostRotation))

        # Construct scaling matrix
        lScaling = node.LclScaling.Get()
        lScalingM.SetS(FbxVector4(lScaling))

        # Construct offset and pivot matrices
        lScalingOffset = node.ScalingOffset.Get()
        lScalingPivot = node.ScalingPivot.Get()
        lRotationOffset = node.RotationOffset.Get()
        lRotationPivot = node.RotationPivot.Get()
        lScalingOffsetM.SetT(FbxVector4(lScalingOffset))
        lScalingPivotM.SetT(FbxVector4(lScalingPivot))
        lRotationOffsetM.SetT(FbxVector4(lRotationOffset))
        lRotationPivotM.SetT(FbxVector4(lRotationPivot))

        # Calculate the global transform matrix of the parent node
        lParentNode = node.GetParent()
        if(lParentNode):
            lParentGX = self.CalculateGlobalTransform(lParentNode)
        else:
            lParentGX.SetIdentity()
        
        # Construct Global Rotation
        lLRM = FbxAMatrix()
        lParentGRM = FbxAMatrix()
        lParentGR = lParentGX.GetR()
        lParentGRM.SetR(lParentGR)
        lLRM = lPreRotationM * lRotationM * lPostRotationM

        # Construct Global Shear*Scaling
        # FBX SDK does not support shear, to patch this, we use:
        # Shear*Scaling = RotationMatrix.Inverse * TranslationMatrix.Inverse * WholeTranformMatrix
        lLSM = FbxAMatrix()
        lParentGSM = FbxAMatrix() 
        lParentGRSM = FbxAMatrix()
        lParentTM = FbxAMatrix()
        lParentGT = lParentGX.GetT()
        lParentTM.SetT(lParentGT)
        lParentGRSM = lParentTM.Inverse() * lParentGX
        lParentGSM = lParentGRM.Inverse() * lParentGRSM
        lLSM = lScalingM

        # Do not consider translation now
        lInheritType = node.InheritType.Get()
        if lInheritType == FbxTransform.eInheritRrSs:
            lGlobalRS = lParentGRM * lLRM * lParentGSM * lLSM
        elif lInheritType == FbxTransform.eInheritRSrs:
            lGlobalRS = lParentGRM * lParentGSM * lLRM * lLSM
        elif(lInheritType == FbxTransform.eInheritRrs):
            lParentLSM = FbxAMatrix()
            lParentLS = lParentNode.LclScaling.Get()
            lParentLSM.SetS(FbxVector4(lParentLS))
            lParentGSM_noLocal = lParentGSM * lParentLSM.Inverse()
            lGlobalRS = lParentGRM * lLRM * lParentGSM_noLocal * lLSM
        else:
            print("error, unknown inherit type! \n")

        OT = lRotationOffsetM * lRotationPivotM * lPreRotationM * lRotationM * lPostRotationM * lRotationPivotM.Inverse()\
            * lScalingOffsetM * lScalingPivotM * lScalingM * lScalingPivotM.Inverse()
        lTransform = lTranlationM * OT
        lLocalTWithAllPivotAndOffsetInfo = lTransform.GetT()
        
        # Calculate global translation vector according to: 
        # GlobalTranslation = ParentGlobalTransform * LocalTranslationWithPivotAndOffsetInfo
        lGlobalTranslation = lParentGX.MultT(lLocalTWithAllPivotAndOffsetInfo)
        lGlobalT.SetT(lGlobalTranslation)

        # Construct the whole global transform
        lTransform = lGlobalT * lGlobalRS

        # D = FbxAMatrix()
        # D.SetT(FbxVector4(5.0, 5.0, 5.0))
        lTransform = lTransform
        lTransform = lParentGX.Inverse() * D * lTransform * lGlobalRS.Inverse()
        lTranlationM = lTransform * OT.Inverse()
        D = lTranlationM.GetT()
        node.LclTranslation.Set(FbxDouble3(D[0], D[1], D[2]))

    def transfer(self, fbx_class, target_vertices=None, target_keypoints=None):
        count = fbx_class.scene.GetPoseCount()
        for _ in range(count):
            fbx_class.scene.RemovePose(0)

        # get template mesh
        template = fbx_class.get_node_by_name("unamed").GetMesh()

        # update mesh's vertices
        i = 0
        for v in template.GetControlPoints():
            v = FbxVector4(target_vertices[i][0], target_vertices[i][1], target_vertices[i][2])
            template.SetControlPointAt(v, i)
            i += 1

        # check the modified model
        polygonVertices = np.array(template.GetPolygonVertices()).reshape(-1, 4)
        vertex_cout = np.array(template.GetControlPointsCount())
        check_vertices = []
        i = 0
        for v in template.GetControlPoints():
            check_vertices.append([v[0], v[1], v[2]])
            i += 1
        out = om.PolyMesh(points=check_vertices, face_vertex_indices=polygonVertices)
        om.write_mesh("model.obj", out)

        root_node = fbx_class.root_node
        skeletons = []
        skeletons.append([(root_node.GetName(), root_node)])
        current = 0
        count = 0
        while count < 25:
            parents = skeletons[current]
            skeletons.append([])
            for _, p in parents:
                for c in range(p.GetChildCount()):
                    if p.GetChild(c).GetName() != "unamed":
                        skeletons[current+1].append((p.GetChild(c).GetName(), p.GetChild(c)))
                        count += 1
            current += 1
            if len(skeletons[-1])==0:
                if 0:
                    print("cannot reach requirement ",count)
                    for i in range(len(skeletons)):
                        print("--level--",i)
                        for x in skeletons[i]:
                            print(x[0])
                break
        '''
        data = {}
        points = []
        for s in range(1, len(skeletons)):
            for _, node in skeletons[s]:
                p = np.array([0.0, 0.0, 0.0, 1.0])
                t2 = update(node.GetName(), node)
                M = []
                for i in range(4):
                    r = t2.GetRow(i)
                    M.append([r[0], r[1], r[2], r[3]])
                M = np.array(M).T
                p = np.matmul(M, p)
                points.append(p[0:3])
                data[node.GetName()] = M.T
        skeleton = om.PolyMesh(points=points)
        om.write_mesh("skeleton9.obj", skeleton)
        '''
        
        for s in range(1, len(skeletons)):
            for _, node in skeletons[s]:
                
                '''
                print("G1*********************")
                for r in data[node.GetName()]:
                    print(r[0], r[1], r[2], r[3])
                '''

                parent = self.CalculateGlobalTransform(node.GetParent())
                source = self.CalculateGlobalTransform(node).GetT()
                try:
                    target = target_keypoints[node.GetName()]
                except:
                    print("miss",node.GetName())
                    continue
                D = FbxAMatrix()
                D.SetT(FbxVector4(target[0] - source[0], target[1] - source[1], target[2] - source[2]))
                self.modify(node, parent, D)
                t = self.CalculateGlobalTransform(node)

                """
                print("G1*********************")
                for i in range(4):
                    r = t.GetRow(i)
                    print(r)
                print("\n")
                """ 

        deformer = template.GetDeformer(0)
        clusters = {}
        for i in range(deformer.GetClusterCount()):
            cluster = deformer.GetCluster(i)
            link = cluster.GetLink()
            clusters[link.GetName()] = cluster     
        for name, cluster in clusters.items():
            t = cluster.GetLink().EvaluateGlobalTransform()
            cluster.SetTransformLinkMatrix(t)
        
        points = []
        for s in range(1, len(skeletons)):
            for _, node in skeletons[s]:
                p = np.array([0.0, 0.0, 0.0, 1.0])
                t = node.EvaluateGlobalTransform()
                p = t.GetT()
                points.append([p[0], p[1], p[2]])
        skeleton = om.PolyMesh(points=points)
        om.write_mesh("skeleton.obj", skeleton)

if __name__ == '__main__':
    import sys
    sys.path.append('FBX20190_FBXFILESDK_LINUX/lib/Python27_ucs4_x64')
    import openmesh as om
    from FBXClass import FBXClass

    target_keypoints = {'mixamorig:LeftLeg': [0.19146551263092818, -0.6540266315228804, -0.021430224208800204], 'mixamorig:Neck': [0.0, 0.35245993175227125, -0.06565949496487877], 'mixamorig:Spine1': [0.0, -0.03232073178702613, -0.03774766667872774], 'mixamorig:Spine2': [0.0, 0.14875257947165454, -0.05088264670829076], 'mixamorig:LeftFoot': [0.16441401812875045, -0.8232710927100443, -0.05519229176502708], 'mixamorig:RightShoulder': [-0.1027444452047348, 0.3347580079050728, -0.06596835133294364], 'mixamorig:LeftShoulder': [0.1027444452047348, 0.3347580079050728, -0.06535064604739448], 'mixamorig:Spine': [0.0, -0.1907597929239273, -0.0262545607984066], 'mixamorig:RightToeBase': [-0.1723526283150312, -0.9574540672352234, 0.06980648830566419], 'mixamorig:RightUpLeg': [-0.17054346203804016, -0.402091920375824, -0.021858789026737213], 'mixamorig:LeftForeArm': [0.4975248087512829, 0.21623093522038356, -0.015447707586599323], 'mixamorig:RightLeg': [-0.1914655119179155, -0.6540266208859554, -0.022460974737139865], 'mixamorig:RightForeArm': [-0.49752486933583195, 0.21623112930101535, -0.015600770035602174], 'mixamorig:LeftToeBase': [0.1723521112069892, -0.9574535732630359, 0.06710000990297416], 'mixamorig:LeftUpLeg': [0.17054346203804016, -0.402091920375824, -0.019634637981653214], 'mixamorig:LeftArm': [0.308255570237736, 0.29971006140746553, -0.06473293626398335], 'mixamorig:LeftHand': [0.6588356461460728, 0.17191041033208426, 0.028116072259702807], 'mixamorig:RightFoot': [-0.16441377996487191, -0.8232727579656831, -0.05414916005946043], 'mixamorig:LeftToe_End': [0.17294317431393672, -0.9568778358323612, 0.140106626073431], 'mixamorig:Head': [0.0, 0.4651883784743019, -0.041857415704529506], 'mixamorig:Hips': [0.0, -0.3265646994113922, -0.016403328627347946], 'mixamorig:RightArm': [-0.308255587374755, 0.29971016187105426, -0.0665860598038212], 'mixamorig:HeadTop_End': [0.0, 0.9666539672823615, 0.06402471485872957], 'mixamorig:RightToe_End': [-0.17294407875965895, -0.956877578617603, 0.1436928574474825], 'mixamorig:RightHand': [-0.6588357289638186, 0.17191068569591544, 0.028090579230675324]}
    target = om.read_polymesh("target3.OBJ")
    target_vertices = target.points()
    # I can not find a deep-copy in the fbx sdk, so I re-read it in each iteration
    fbx_class = FBXClass('source.fbx')
    s = SkeletonTransfer()
    s.transfer(fbx_class, target_vertices, target_keypoints)
