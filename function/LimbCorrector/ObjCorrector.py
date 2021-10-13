import numpy as np
import meshio
import openmesh as om
import pickle
import pdb
class ObjCorrector:
	def __init__(self):
		dataroot = ""
		clusterdic = np.load(dataroot + 'clusterdic.npy' , allow_pickle=True).item()
		self.index2cluster = {}
		for key in clusterdic.keys():
			val = clusterdic[key]
			self.index2cluster[val] = key

		self.joint2index = np.load(dataroot + 'joint2index.npy' , allow_pickle=True).item()
		self.ktree_table = np.load(dataroot + 'ktree_table.npy' , allow_pickle=True)
		self.weightMatrix = np.load(dataroot + 'weightMatrix.npy' , allow_pickle=True)
		self.parent = self.ktree_table

	def rotation_matrix_from_vectors(self,vec1, vec2):
			""" Find the rotation matrix that aligns vec1 to vec2
			:param vec1: A 3d "source" vector
			:param vec2: A 3d "destination" vector
			:return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
			"""
			a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
			v = np.cross(a, b)
			c = np.dot(a, b)
			s = np.linalg.norm(v)
			# print(s)
			kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
			rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
			return rotation_matrix

	def with_zeros(self,x):
		return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

	def pack(self,x):
		return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

	def Getrotation(self,Jp,Js,target=[1,0,0]):
		source = Js -Jp
		dis = np.linalg.norm(source)
		tar = dis*np.array(target)
		return self.rotation_matrix_from_vectors(source,tar)

	def run(self,filename="s.obj",targetname="re.obj",check=False):
		mesh = om.read_polymesh(filename)
		points = mesh.points()
		faces = mesh.face_vertex_indices()

		J = []
		for i in range(len(self.index2cluster)):
			key = self.index2cluster[i]
			if key =='RootNode':
				J.append(np.array([0,0,0]))
				continue
			index_list = self.joint2index[key]
			index_val = []
			for index in index_list:
				index_val.append(points[index])
			index_val = np.array(index_val)
			maxval = index_val.max(0)
			minval = index_val.min(0)
			J.append((maxval+minval)*1.0/2)
		J = np.array(J)

		G = np.empty((self.ktree_table.shape[0], 4, 4))

		Rset = {}

		#6 arm #17
		#7 forearm #18
		#8 hand #19
		# print('--old--')
		# print(J[7]-J[6],J[8]-J[7])
		limb = [[6,7,8,[1,0,0]],
						[17,18,19,[-1,0,0]],
						[12,20,21,[0,-1,0]],
						[11,13,14,[0,-1,0]]]
		for a,b,c,direct in limb:
			Rparent = self.Getrotation(J[a],J[b],target=direct)
			Rset[a] = Rparent
			Rset[b] = self.Getrotation(J[b].dot(Rparent.T),J[c].dot(Rparent.T),target=direct)
			Rset[c] = Rset[b].T
		if check:
			print("---old---")
			for a,b,c,direct in limb:
				print(J[b]-J[a],J[c]-J[b],direct)




		G[0] = np.eye(4)
		for i in range(1, self.ktree_table.shape[0]):
			if i in Rset:
				R = Rset[i]
			else:
				R = np.eye(3)
				
			G[i] = G[int(self.parent[i])].dot(self.with_zeros(
					np.hstack(
						[R,((J[i, :]-J[int(self.parent[i]),:]).reshape([3,1]))]
					)
				)
			)
		# remove the transformation due to the rest pose

		G = G - self.pack(
			np.matmul(
				G,
				np.hstack([J, np.zeros([23, 1])]).reshape([23, 4, 1])
				)
			)
				# transformation of each vertex
		v_posed = points
		rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
		T = np.tensordot(self.weightMatrix, G, axes=[[1], [0]])
		v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]

		newmesh = om.PolyMesh(points=v ,face_vertex_indices=faces)
		om.write_mesh(targetname,newmesh)


		if check:
		
			J = []
			for i in range(len(self.index2cluster)):
				key = self.index2cluster[i]
				if key =='RootNode':
					J.append(np.array([0,0,0]))
					continue
				index_list = self.joint2index[key]
				index_val = []
				for index in index_list:
					index_val.append(v[index])
				index_val = np.array(index_val)
				maxval = index_val.max(0)
				minval = index_val.min(0)
				J.append((maxval+minval)*1.0/2)
			J = np.array(J)
			# print(J[6],J[7],J[8])

			# print('--new--')
			# print(J[7]-J[6],J[8]-J[7])
			print("---new---")
			for a,b,c,direct in limb:
				print(J[b]-J[a],J[c]-J[b],direct)



