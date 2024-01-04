import pygame as pg
import pyrr
import copy
import cv2
from PIL import Image
import time
import multiprocessing
from scipy.io import loadmat
from numba import cuda
import numpy as np
import numpy as np
# import cupy
from numba import jit, njit, prange
import time
from numba import cuda
import cupy as cp
from itertools import product
from scipy.io import loadmat, savemat
import numpy as np
import math
import copy
# from PIL import Image
from numpy import cos, sin, sqrt
from math import cos, sin, atan2, asin
from scipy.signal import savgol_filter
from numpy import cos, sin, sqrt
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GL import *
@cuda.jit
def runner(p1, gri, out_arr, grid2, grid3,n0,n1,n2,diff_arr,expr_arr):
    i, j = cuda.grid(2)
    if i < gri.shape[0] and j < gri.shape[1]:
        c = np.cos(gri[i, j])
        t = 1 - c
        s = np.sin(gri[i, j])
        X = n0
        Y = n1
        Z = n2
        d11 = t * X ** 2 + c
        d12 = t * X * Y - s * Z
        d13 = t * X * Z + s * Y
        d21 = t * X * Y + s * Z
        d22 = t * Y ** 2 + c
        d23 = t * Y * Z - s * X
        d31 = t * X * Z - s * Y
        d32 = t * Y * Z + s * X
        d33 = t * Z ** 2 + c
        for kin in range(638324):
            out_arr[i, j, kin, 0] = d11 * diff_arr[kin, 0] + d12 * diff_arr[kin, 1] + d13 * diff_arr[kin, 2]
            out_arr[i, j, kin, 0] = out_arr[i, j, kin, 0] + p1[0]
            out_arr[i, j, kin, 1] = d21 * diff_arr[kin, 0] + d22 * diff_arr[kin, 1] + d23 * diff_arr[kin, 2]
            out_arr[i, j, kin, 1] = out_arr[i, j, kin, 1] + p1[1]
            out_arr[i, j, kin, 2] = d31 * diff_arr[kin, 0] + d32 * diff_arr[kin, 1] + d33 * diff_arr[kin, 2]
            out_arr[i, j, kin, 2] = out_arr[i, j, kin, 2] + p1[2]
        N0 = out_arr[i, j, 638321, 0] - out_arr[i, j, 638320, 0]
        N1 = out_arr[i, j, 638321, 1] - out_arr[i, j, 638320, 1]
        N2 = out_arr[i, j, 638321, 2] - out_arr[i, j, 638320, 2]
        Nm = (N0 ** 2 + N1 ** 2 + N2 ** 2) ** 0.5
        N0 =  N0 / Nm
        N1 = N1 / Nm
        N2 = N2 / Nm
        c = np.cos(grid2[i, j])
        t = 1 - c
        s = np.sin(grid2[i, j])
        X = N0
        Y = N1
        Z = N2
        d11 = t * X ** 2 + c
        d12 = t * X * Y - s * Z
        d13 = t * X * Z + s * Y
        d21 = t * X * Y + s * Z
        d22 = t * Y ** 2 + c
        d23 = t * Y * Z - s * X
        d31 = t * X * Z - s * Y
        d32 = t * Y * Z + s * X
        d33 = t * Z ** 2 + c
        for kin in range(638324):
            expr_arr[i,j,kin,0] = out_arr[i, j, kin, 0] - out_arr[i, j, 638320, 0]
            expr_arr[i, j, kin, 1] = out_arr[i, j, kin, 1] - out_arr[i, j, 638320, 1]
            expr_arr[i, j, kin, 2] = out_arr[i, j, kin, 2] - out_arr[i, j, 638320, 2]
        base_value0 = out_arr[i, j, 638320, 0]
        base_value1 = out_arr[i, j, 638320, 1]
        base_value2 = out_arr[i, j, 638320, 2]
        for kin in range(638324):
           expr1 = expr_arr[i,j,kin,0]
           expr2 = expr_arr[i, j, kin, 1]
           expr3 = expr_arr[i, j, kin, 2]
           out_arr[i, j, kin, 0] = d11 *  expr1   + d12 * expr2+ d13 * expr3
           out_arr[i, j, kin, 1] = d21 * expr1 + d22 * expr2  + d23 * expr3
           out_arr[i, j, kin, 2] = d31 * expr1  + d32 * expr2  + d33 * expr3
           out_arr[i, j, kin, 0] = out_arr[i, j, kin, 0] + base_value0
           out_arr[i, j, kin, 1] = out_arr[i, j, kin, 1] + base_value1
           out_arr[i, j, kin, 2] = out_arr[i, j, kin, 2] + base_value2
        N0 = out_arr[i, j, 638323, 0] - out_arr[i, j, 638322, 0]
        N1 = out_arr[i, j, 638323, 1] - out_arr[i, j, 638322, 1]
        N2 = out_arr[i, j, 638323, 2] - out_arr[i, j, 638322, 2]
        Nm = (N0 ** 2 + N1 ** 2 + N2 ** 2) ** 0.5
        N0 = N0 / Nm
        N1 = N1 / Nm
        N2 = N2 / Nm
        c = np.cos(grid3[i, j])
        t = 1 - c
        s = np.sin(grid3[i, j])
        X = N0
        Y = N1
        Z = N2
        d11 = t * X ** 2 + c
        d12 = t * X * Y - s * Z
        d13 = t * X * Z + s * Y
        d21 = t * X * Y + s * Z
        d22 = t * Y ** 2 + c
        d23 = t * Y * Z - s * X
        d31 = t * X * Z - s * Y
        d32 = t * Y * Z + s * X
        d33 = t * Z ** 2 + c
        for kin in range(638324):
            expr_arr[i, j, kin, 0] = out_arr[i, j, kin, 0] - out_arr[i, j, 638322, 0]
            expr_arr[i, j, kin, 1] = out_arr[i, j, kin, 1] - out_arr[i, j, 638322, 1]
            expr_arr[i, j, kin, 2] =  out_arr[i, j, kin, 2] - out_arr[i, j, 638322, 2]
        base_value0 = out_arr[i, j, 638322, 0]
        base_value1 = out_arr[i, j, 638322, 1]
        base_value2 = out_arr[i, j, 638322, 2]
        for kin in range(638324):
           expr1 = expr_arr[i, j, kin, 0]
           expr2 = expr_arr[i, j, kin, 1]
           expr3 = expr_arr[i, j, kin, 2]
           out_arr[i, j, kin, 0] = d11 * expr1 + d12 * expr2 + d13 * expr3
           out_arr[i, j, kin, 1] = d21 * expr1 + d22 * expr2 + d23 * expr3
           out_arr[i, j, kin, 2] = d31 * expr1 + d32 * expr2 + d33 * expr3
           out_arr[i, j, kin, 0] = out_arr[i, j, kin, 0] + base_value0
           out_arr[i, j, kin, 1] = out_arr[i, j, kin, 1] + base_value1
           out_arr[i, j, kin, 2] = out_arr[i, j, kin, 2] + base_value2
def PointRotate3DMineO(p1, p2, p0, theta):
    arrn = np.zeros(shape=(638320, 3), dtype=np.float16)
    arrn[:, 0] = p1[0]
    arrn[:, 1] = p1[1]
    arrn[:, 2] = p1[2]
    p = np.subtract(p0, arrn)
    q = np.array([0.0, 0.0, 0.0], dtype=np.float16)
    N = (p2 - p1)
    Nm = sqrt(N[0] ** 2 + N[1] ** 2 + N[2] ** 2)
    n = np.array([N[0] / Nm, N[1] / Nm, N[2] / Nm], dtype=np.float16)
    c = cos(theta)
    t = (1 - cos(theta))
    s = sin(theta)
    X = n[0]
    Y = n[1]
    Z = n[2]
    d11 = t * X ** 2 + c
    d12 = t * X * Y - s * Z
    d13 = t * X * Z + s * Y
    d21 = t * X * Y + s * Z
    d22 = t * Y ** 2 + c
    d23 = t * Y * Z - s * X
    d31 = t * X * Z - s * Y
    d32 = t * Y * Z + s * X
    d33 = t * Z ** 2 + c
    q = np.zeros(shape=(638320, 3), dtype=np.float16)
    q[:, 0] = d11 * p[:, 0] + d12 * p[:, 1] + d13 * p[:, 2]
    q[:, 1] = d21 * p[:, 0] + d22 * p[:, 1] + d23 * p[:, 2]
    q[:, 2] = d31 * p[:, 0] + d32 * p[:, 1] + d33 * p[:, 2]
    return q + arrn
def load_voa_vbos(vertices):
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    return vao, vbo
def loadMesh(filepath):
    v = []
    vt = []
    vn = []
    vertices = []
    with open(filepath, "r") as f:
        line = f.readline()
        while line:
            firstSpace = line.find(" ")
            flag = line[0:firstSpace]
            if flag == "v":
                line = line.split(" ")
                v.append([float(line[1]), float(line[2]), float(line[3])])
            elif flag == "vt":
                line = line.split(" ")
                vt.append([float(line[1]), float(line[2])])
            elif flag == "vn":
                line = line.split(" ")
                vn.append([float(line[1]), float(line[2]), float(line[3])])
            elif flag == "f":
                line = line.replace("f ", "")
                line = line.replace("\n", "")
                line = line.split(" ")
                face_Vertices = []
                face_Textures = []
                face_Normals = []
                for vertex in line:
                    l = vertex.split("/")
                    position = int(l[0]) - 1
                    face_Vertices.append(position)
                    texture = int(l[1]) - 1
                    face_Textures.append([vt[texture][0], 1 - vt[texture][1]])
                    normal = int(l[2]) - 1
                    face_Normals.append(vn[normal])
                triangles_in_face = len(line) - 2
                vertex_order = []
                for i in range(triangles_in_face):
                    vertex_order.append(0)
                    vertex_order.append(i + 1)
                    vertex_order.append(i + 2)
                for i in vertex_order:
                    vertices.append(face_Vertices[i])
                    vertices.append(0)
                    vertices.append(0)
                    for x in face_Textures[i]:
                        vertices.append(x)
                    for x in face_Normals[i]:
                        vertices.append(x)
            line = f.readline()
    return vertices
def loadMeshN(filepath):
    v = []
    vt = []
    vn = []
    with open(filepath, "r") as f:
        line = f.readline()
        while line:
            firstSpace = line.find(" ")
            flag = line[0:firstSpace]
            if flag == "v":
                line = line.split(" ")
                v.append([float(line[1]), float(line[2]), float(line[3])])
            elif flag == "vt":
                line = line.split(" ")
                vt.append([float(line[1]), float(line[2])])
            elif flag == "vn":
                line = line.split(" ")
                vn.append([float(line[1]), float(line[2]), float(line[3])])
            line = f.readline()
    vertices = []
    mtl_count = 0
    with open(filepath, "r") as f:
        line = f.readline()
        while line:
            firstSpace = line.find(" ")
            flag = line[0:firstSpace]
            if flag == "f":
                line = line.replace("f ", "")
                line = line.replace("\n", "")
                line = line.split(" ")
                face_Vertices = []
                face_Textures = []
                face_Normals = []
                for vertex in line:
                    l = vertex.split("/")
                    position = int(l[0]) - 1
                    face_Vertices.append(position)
                    texture = int(l[1]) - 1
                    face_Textures.append([vt[texture][0], 1 - vt[texture][1]])
                    normal = int(l[2]) - 1
                    face_Normals.append(vn[normal])
                triangles_in_face = len(line) - 2
                vertex_order = []
                for i in range(triangles_in_face):
                    vertex_order.append(0)
                    vertex_order.append(i + 1)
                    vertex_order.append(i + 2)
                for i in vertex_order:
                    vertices.append(face_Vertices[i])
                    vertices.append(0)
                    vertices.append(0)
                    for x in face_Textures[i]:
                        vertices.append(x)
                    for x in face_Normals[i]:
                        vertices.append(x)
            elif flag == "usemtl":
                if mtl_count == 0:
                    mtl_count += 1
                    pass
                elif mtl_count == 1:
                    vertices1 = vertices
                    vertices = []
                    mtl_count += 1
                elif mtl_count == 2:
                    vertices2 = vertices
                    vertices = []
                    mtl_count += 1
            line = f.readline()
    vertices3 = vertices
    return vertices1, vertices2, vertices3
class Cube:
    def __init__(self, position, eulers):
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)

class App:
    def __init__(self,STARTX,STARTY,ENDX,ENDY,SIZEXY):
        pg.init()
        pg.display.set_mode((2000, 2000), pg.OPENGL | pg.DOUBLEBUF | pg.HWSURFACE | pg.NOFRAME)
        self.clock = pg.time.Clock()
        glClearColor(1, 1, 1, 1)
        glEnable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.shader = self.create_shader("shaders/vertex.txt", "shaders/fragment.txt")
        glUseProgram(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, "imageTexture"), 0)
        self.cube = Cube(position=[0, 0, -10], eulers=[0, 0, 0])
        self.vertices1, self.vertices2, self.vertices3 = loadMeshN(
            r"G:\0_0\image_folder\gray\relevant_data\head-03-a\source\head 01 a/HEAD 03 AW8.obj")
        self.vertices1 = np.array(self.vertices1, dtype=np.float32)
        self.vertices2 = np.array(self.vertices2, dtype=np.float32)
        self.vertices3 = np.array(self.vertices3, dtype=np.float32)
        self.wood_texture1 = Material(
            r"G:\0_0\image_folder\gray\relevant_data\head-03-a\source\head 01 a/HEAD 03 A.jpg")
        self.wood_texture2 = Material(
            r"G:\0_0\image_folder\gray\relevant_data\head-03-a\source\head 01 a/HEAD 03 A1.jpg")
        self.wood_texture3 = Material(
            r"G:\0_0\image_folder\gray\relevant_data\head-03-a\source\head 01 a/HEAD 03 A2.jpg")
        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy=45, aspect=2000 / 2000,
            near=0.1, far=100, dtype=np.float32)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, projection_transform)
        self.modelMatrixLocation = glGetUniformLocation(self.shader, "model")
        self.mainLoop(STARTX,STARTY,ENDX,ENDY,SIZEXY)
    def create_shader(self, vertex_filepath, fragment_filepath):
        with open(vertex_filepath, 'r') as f:
            vertex_src = f.readlines()
        with open(fragment_filepath, 'r') as f:
            fragment_src = f.readlines()
        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),compileShader(fragment_src, GL_FRAGMENT_SHADER))
        return shader
    def mainLoop(self,STARTX,STARTY,ENDX,ENDY,SIZEXY):
        ult_copy1 = copy.deepcopy(self.vertices1).astype(int)
        ult_copy2 = copy.deepcopy(self.vertices2).astype(int)
        ult_copy3 = copy.deepcopy(self.vertices3).astype(int)
        glUseProgram(self.shader)
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_eulers(
                eulers=np.radians(self.cube.eulers),
                dtype=np.float32))
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_translation(
                vec=self.cube.position,
                dtype=np.float32))
        glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, model_transform)
        mat = loadmat(r"G:\0_0\image_folder\gray\relevant_data\head-03-a\source\head 01 a/matty.mat")
        ver = mat['vertices'].astype(np.float16)
        XLAS = ver[:, 0]
        YLAS = ver[:, 1]
        ZLAS = ver[:, 2]
        xmin = np.min(XLAS)
        xmax = np.max(XLAS)
        ymin = np.min(YLAS)
        ymax = np.max(YLAS)
        zmin = np.min(ZLAS)
        zmax = np.max(ZLAS)
        zmid = (zmin + zmax) / 2
        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2
        p1 = np.array([xmid - 5, ymid, zmid], dtype=np.float16)
        p2 = np.array([xmid + 5, ymid, zmid], dtype=np.float16)
        theta = np.radians(-90)
        vers = mat['vertices'].astype(np.float16)
        arr = np.zeros(shape=(638320, 3), dtype=np.float16)
        arr[:, 0] = vers[:, 0]
        arr[:, 1] = vers[:, 1]
        arr[:, 2] = vers[:, 2]
        p0 = arr
        pout = PointRotate3DMineO(p1, p2, p0, theta)
        mat['vertices'][:, 0] = pout[:, 0]
        mat['vertices'][:, 1] = pout[:, 1]
        mat['vertices'][:, 2] = pout[:, 2]
        k = 5
        xmid = -1.5497165
        ymid = 2.457754500000002
        zmid = -16.007714500000002
        p1 = np.array([xmid, ymid - 15, zmid], dtype=np.float16)
        p2 = np.array([xmid, ymid + 15, zmid], dtype=np.float16)
        vers = mat['vertices']
        p0 = np.zeros(shape=(638324, 3), dtype=np.float16)
        p0[:638320, 0] = vers[:, 0]
        p0[:638320, 1] = vers[:, 1]
        p0[:638320, 2] = vers[:, 2]
        xmid = -1.1497165
        ymid = 2.457754500000002
        zmid = -16.007714500000002
        pX1 = [xmid - 15, ymid, zmid]
        pX2 = [xmid + 15, ymid, zmid]
        p0[638320] = pX1
        p0[638321] = pX2
        xmid = -1.1497165
        ymid = 2.457754500000002
        zmid = -16.007714500000002
        pZ1 = [xmid, ymid, zmid - 15]
        pZ2 = [xmid, ymid, zmid + 15]
        p0[638322] = pZ1
        p0[638323] = pZ2
        yaw_name = []
        pitch_name = []
        roll_name = []
        for i in range(-90, 90, 1):
            yaw_name.append(i)
        for i in range(-40, 60, 1):
            pitch_name.append(i)
        for i in range(-40, 50, 1):
            roll_name.append(i)
        name_combinations = list(product(yaw_name, pitch_name, roll_name))
        name_combinations = np.array(name_combinations, dtype=np.int8)
        name_combinations = np.reshape(name_combinations[:1617984, :], newshape=(1272, 1272, 3))
        yaw_range = []
        pitch_range = []
        roll_range = []
        for i in range(-90, 90, 1):
            yaw_range.append(np.radians(i))
        for i in range(-40, 60, 1):
            pitch_range.append(np.radians(i))
        for i in range(-40, 50, 1):
            roll_range.append(np.radians(i))
        combinations = list(product(yaw_range, pitch_range, roll_range))
        combinations = np.array(combinations, dtype=np.float32)
        combinations = np.reshape(combinations[:1617984, :], newshape=(1272, 1272, 3))
        combinations = np.ascontiguousarray(combinations)
        combinations = cuda.to_device(combinations)
        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(SIZEXY / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(SIZEXY / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        diff_arr = np.zeros(shape=(638324, 3), dtype=np.float16)
        for kin in range(638324):
            diff_arr[kin, 0] = p0[kin, 0] - p1[0]
            diff_arr[kin, 1] = p0[kin, 1] - p1[1]
            diff_arr[kin, 2] = p0[kin, 2] - p1[2]
        diff_arr = cuda.to_device(diff_arr)
        p1 = cuda.to_device(p1)
        p2 = cuda.to_device(p2)
        p0 = cuda.to_device(p0)
        N0 = p2[0] - p1[0]
        N1 = p2[1] - p1[1]
        N2 = p2[2] - p1[2]
        Nm = (N0 ** 2 + N1 ** 2 + N2 ** 2) ** 0.5
        n0 = N0 / Nm
        n1 = N1 / Nm
        n2 = N2 / Nm
        q = np.zeros(shape=(638324, 3), dtype=np.float16)
        q = cuda.to_device(q)
        p = np.zeros(shape=(638324, 3), dtype=np.float16)
        p = cuda.to_device(p)
        out_arr = np.zeros(shape=(SIZEXY, SIZEXY, 638324, 3), dtype=np.float16)
        out_arr = cuda.to_device(out_arr)
        expr_arr = np.zeros(shape=(SIZEXY, SIZEXY, 638324, 3), dtype=np.float16)
        expr_arr = cuda.to_device(expr_arr)
        for kj in range(STARTY, ENDY, SIZEXY):
           for ki in range(STARTX, ENDX, SIZEXY):
             end_pointy = kj + SIZEXY
             end_pointx = ki + SIZEXY
             if end_pointx>ENDX:
                 end_pointx = ENDX
             if end_pointy>ENDY:
                 end_pointy = ENDY
             namecombs = name_combinations[kj:end_pointy, ki:end_pointx, :]
             grid = combinations[kj:end_pointy, ki:end_pointx, 0]
             grid2 = combinations[kj:end_pointy, ki:end_pointx, 1]
             grid3 = combinations[kj:end_pointy, ki:end_pointx, 2]
             blockspergrid_x = int(np.ceil(grid.shape[0] / threadsperblock[0]))
             blockspergrid_y = int( np.ceil(grid.shape[1]/threadsperblock[1]))
             blockspergrid =  (blockspergrid_x,blockspergrid_y)
             runner[blockspergrid, threadsperblock](p1, grid, out_arr, grid2, grid3, n0, n1, n2, diff_arr, expr_arr)
             out_arro = out_arr.copy_to_host()
             FULL_MAT = out_arro
             for counti in range(namecombs.shape[0]):
                 for countj in range(namecombs.shape[1]):
                     matfile = FULL_MAT[counti, countj]
                     inds_arr = ult_copy1[:ult_copy1.size:8]
                     vert_arr = matfile[inds_arr]
                     vert_arrx = vert_arr[:, 0]
                     vert_arry = vert_arr[:, 1]
                     vert_arrz = vert_arr[:, 2]
                     self.vertices1[:self.vertices1.size:8] = vert_arrx
                     self.vertices1[1:self.vertices1.size:8] = vert_arry
                     self.vertices1[2:self.vertices1.size:8] = vert_arrz
                     self.vertices1 = self.vertices1.astype(np.float32)
                     self.vertex_count = len(self.vertices1) // 8
                     voa, vbo = load_voa_vbos(self.vertices1)
                     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                     self.wood_texture1.use()
                     glBindVertexArray(voa)
                     glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
                     inds_arr = ult_copy2[:ult_copy2.size:8]
                     vert_arr = matfile[inds_arr]
                     vert_arrx = vert_arr[:, 0]
                     vert_arry = vert_arr[:, 1]
                     vert_arrz = vert_arr[:, 2]
                     self.vertices2[:self.vertices2.size:8] = vert_arrx
                     self.vertices2[1:self.vertices2.size:8] = vert_arry
                     self.vertices2[2:self.vertices2.size:8] = vert_arrz
                     self.vertices2 = self.vertices2.astype(np.float32)
                     self.vertex_count = len(self.vertices2) // 8
                     glDeleteVertexArrays(1, [voa])
                     glDeleteBuffers(1, [vbo])
                     voa, vbo = load_voa_vbos(self.vertices2)
                     self.wood_texture2.use()
                     glBindVertexArray(voa)
                     glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
                     inds_arr = ult_copy3[:ult_copy3.size:8]
                     vert_arr = matfile[inds_arr]
                     vert_arrx = vert_arr[:, 0]
                     vert_arry = vert_arr[:, 1]
                     vert_arrz = vert_arr[:, 2]
                     self.vertices3[:self.vertices3.size:8] = vert_arrx
                     self.vertices3[1:self.vertices3.size:8] = vert_arry
                     self.vertices3[2:self.vertices3.size:8] = vert_arrz
                     self.vertices3 = self.vertices3.astype(np.float32)
                     self.vertex_count = len(self.vertices3) // 8
                     glDeleteVertexArrays(1, [voa])
                     glDeleteBuffers(1, [vbo])
                     voa, vbo = load_voa_vbos(self.vertices3)
                     self.wood_texture3.use()
                     glBindVertexArray(voa)
                     glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
                     pixels = glReadPixels(0, 0, 2000, 2000, GL_RGB, GL_UNSIGNED_BYTE)
                     image_np = np.frombuffer(pixels, dtype=np.uint8).reshape((2000, 2000, 3)).copy()
                     image_np[:, :, [0, 2]] = image_np[:, :, [2, 0]]
                     image_np = np.flip(image_np, axis=0)[320:1120, 475:1275, :]
                     cv2.imwrite(r"I:\DATSET/" + str(namecombs[counti,countj,0]) + "_" + str(namecombs[counti,countj,1]) + "_" +str(namecombs[counti,countj,2]) + ".png", image_np)
                     glDeleteVertexArrays(1, [voa])
                     glDeleteBuffers(1, [vbo])
             del out_arro
    def quit(self):
        # self.cube_mesh.destroy()
        self.wood_texture1.destroy()
        self.wood_texture2.destroy()
        self.wood_texture3.destroy()
        glDeleteProgram(self.shader)
        pg.quit()
class Material:
    def __init__(self, filepath):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        image = pg.image.load(filepath).convert_alpha()
        image_width, image_height = image.get_rect().size
        img_data = pg.image.tostring(image, 'RGBA')
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)
    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
    def destroy(self):
        print(self.texture)
        print((self.texture,))
        glDeleteTextures(1, (self.texture,))


if __name__ == "__main__":
    STARTX = 0
    STARTY = 0
    ENDX = 1272
    ENDY = 636
    SIZEXY = 30
    procc = multiprocessing.Process(target=App, args=(  STARTX,STARTY,ENDX,ENDY,SIZEXY  ))
    STARTX = 0
    STARTY = 636
    ENDX = 1272
    ENDY = 1272
    SIZEXY = 30
    procc2 = multiprocessing.Process(target=App, args=(  STARTX,STARTY,ENDX,ENDY,SIZEXY  ))
    procc.start()
    procc2.start()
    procc.join()
    procc2.join()








   



