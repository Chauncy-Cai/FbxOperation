import numpy as np
import openmesh as om
import pdb

class Transform(object):
    def __init__(self,template_path):
        self.node2color = self.getNode2Color()
        self.color2vertexindex = self.getColor2Index(template_path)
        self.joint2indexlist = {}
    def getNode2Color(self):
        node2color = {}

        '''
        # foot from top to bottom
        node2color['right_thigh'] = (140,53,30)
        node2color['right_knee'] = (90,144,226)
        node2color['right_ankle'] = (77,255,146)
        node2color['right_foot'] = (227,0,197)
        node2color['right_toe'] = (105,227,227)

        node2color['left_thigh'] = (57,96,17)
        node2color['left_knee'] = (228,122,80)
        node2color['left_ankle'] = (69,28,109)
        node2color['left_foot'] = (255,30,0)
        node2color['left_toe'] = (200,177,100)
        
        # hand from left to right
        node2color['right_collar'] = (12,255,255)
        node2color['right_shoulder'] = (154,0,210)
        node2color['right_elbow'] = (250,208,42)
        node2color['right_wrist'] = (250,42,42)
    
        node2color['left_collar'] = (168,255,12)
        node2color['left_shoulder'] = (210,0,118)
        node2color['left_elbow'] = (70,53,247)
        node2color['left_wrist'] = (146,250,42)
        
        # main spine from top to botom
        node2color['head'] = (44,199,199)
        node2color['chin'] = (44,199,199)
        node2color['neck'] = (3,111,22)
        node2color['chest'] = (217,75,78)
        node2color['bottom_costa'] = (140,115,30)
        node2color['navel'] = (210,109,231)
        node2color['cortch'] = (255,84,45)
        '''


        node2color['Lshoulder'] = (154,0,209)
        node2color['Lelbow'] = (249,207,42)
        node2color['Lability'] = (249,42,42)
        node2color['Lhand'] = (222,124,8)
        node2color['Lchest'] = (12,255,255)
        
        # node2color[''] = (223,19,216) # head
        node2color['neck2'] = (43,198,198)
        node2color['neck1'] = (3,110,22)
        node2color['spine3'] = (216,75,77)
        node2color['spine1'] = (209,109,230)
        # node2color['spine0'] = (255,84,45) # useless spine
        node2color['spine0'] = (244,226,151)
        
        node2color['Lass'] = (140,52,29)
        node2color['Lknee'] = (89,144,226)
        node2color['Lankle'] = (77,255,145)
        node2color['Lfoot'] = (226,0,196)

        node2color['Rfoot'] = (255,29,0)
        node2color['Rknee'] = (228,121,79)
        node2color['Rass'] = (56,96,17)
        node2color['Rankle'] = (68,28,109)
        
        node2color['Rchest'] = (168,255,12)
        node2color['Rshoulder'] = (209,0,117)
        node2color['Relbow'] = (70,52,246)
        node2color['Rability'] = (145,249,42)
        node2color['Rhand'] = (91,51,31)
        
        return node2color

    def getColor2Index(self,template_path):
        template_color = om.read_polymesh(template_path,vertex_color=True)

        # get vertex color
        colors = template_color.vertex_colors()
        points = template_color.points()

        color2pos = {}
        color2index = {}
        # calculate the bone node index
        i = 0
        for p,c in zip(points,colors):
            tcolor = c*255
            tcolor = tcolor[:3]
            tcolor = [int(round(x)) for x in tcolor]
            tcolor = tuple(tcolor)
            if tcolor in color2pos:
                color2pos[tcolor].append(p)
                color2index[tcolor].append(i)
            else:
                color2pos[tcolor] = [p]
                color2index[tcolor] = [i]
                # print(tcolor)
            i += 1
        print("There are "+str(len(color2pos))+"colors in total")
        # del noise color
        dellist = []
        for key in color2index:
            if len(color2index[key])<15:
                dellist.append(key)
        for d in dellist:
            del color2index[d]
        del color2index[(255,255,255)]
        return color2index

    def generateSkeleton2Pos(self,obj):
        points = obj.points()
        node2pos = {}
        for node in self.node2color:
            color =  self.node2color[node]
            vertex_index_list = self.color2vertexindex[color]
            pointlist = []
            indexlist = []
            for index in vertex_index_list:
                pointlist.append(points[index])
                indexlist.append(index)
            self.joint2indexlist[node] = indexlist
            pointarray = np.array(pointlist)
            pointmax = pointarray.max(0)
            pointmin = pointarray.min(0)
            pointmean = (pointmax+pointmin)/2.0
            node2pos[node] = pointmean
        return node2pos
        

def check_skeleton(keypointlist):
    skeleton = om.PolyMesh(points=keypointlist)
    om.write_mesh("test.obj",skeleton)
    return

## testing
if __name__ == '__main__':
    template_nocolor = om.read_polymesh("/data/xiaojin/huawei/sources/zhangjing/20210621/cat/1/check/m.OBJ") #target
    Transformer = Transform("template_color.obj") #source 
    dic = Transformer.generateSkeleton2Pos(template_nocolor)
    np.save("joint2index.npy",Transformer.joint2indexlist)
    for key in dic:
        value = dic[key]
        print(key,value)
    check_skeleton(dic.values())
