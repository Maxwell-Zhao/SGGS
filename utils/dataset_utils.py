import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scene.gaussian_model import BasicPointCloud
from plyfile import PlyData, PlyElement
import open3d as o3d
from torch import optim
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

# add ZJUMoCAP dataloader
def get_02v_bone_transforms(Jtr,):
    rot45p = Rotation.from_euler('z', 45, degrees=True).as_matrix()
    rot45n = Rotation.from_euler('z', -45, degrees=True).as_matrix()

    # Specify the bone transformations that transform a SMPL A-pose mesh
    # to a star-shaped A-pose (i.e. Vitruvian A-pose)
    bone_transforms_02v = np.tile(np.eye(4), (24, 1, 1))

    # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    chain = [1, 4, 7, 10]
    rot = rot45p.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
    # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    chain = [2, 5, 8, 11]
    rot = rot45n.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)

    return bone_transforms_02v

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

class AABB(torch.nn.Module):
    def __init__(self, coord_max, coord_min):
        super().__init__()
        self.register_buffer("coord_max", torch.from_numpy(coord_max).float())
        self.register_buffer("coord_min", torch.from_numpy(coord_min).float())

    def normalize(self, x, sym=False):
        x = (x - self.coord_min) / (self.coord_max - self.coord_min)
        if sym:
            x = 2 * x - 1.
        return x

    def unnormalize(self, x, sym=False):
        if sym:
            x = 0.5 * (x + 1)
        x = x * (self.coord_max - self.coord_min) + self.coord_min
        return x

    def clip(self, x):
        return x.clip(min=self.coord_min, max=self.coord_max)

    def volume_scale(self):
        return self.coord_max - self.coord_min

    def scale(self):
        return math.sqrt((self.volume_scale() ** 2).sum() / 3.)


COLOR_MAP_INSTANCES = {
    0: (226., 226., 226.), #(174., 199., 232.),
    1: (120., 94., 240.), #purple
    2: (254., 97., 0.), #orange
    3: (255., 176., 0.), #yellow
    4: (100., 143., 255.), #blue
    5: (220., 38., 127.), #pink
    6: (0., 255., 255.),
    7: (255., 204., 153.),
    8: (255., 102., 0.),
    9: (0., 128., 128.),
    10: (153., 153., 255.),
}

MERGED_BODY_PART_COLORS = {
    0:  (226., 226., 226.),
    1:  (158.0, 143.0, 20.0),  #rightHand
    2:  (243.0, 115.0, 68.0),  #rightUpLeg
    3:  (228.0, 162.0, 227.0), #leftArm
    4:  (210.0, 78.0, 142.0),  #head
    5:  (152.0, 78.0, 163.0),  #leftLeg
    6:  (76.0, 134.0, 26.0),   #leftFoot
    7:  (100.0, 143.0, 255.0), #torso
    8:  (129.0, 0.0, 50.0),    #rightFoot
    9:  (255., 176., 0.),      #rightArm
    10: (192.0, 100.0, 119.0), #leftHand
    11: (149.0, 192.0, 228.0), #rightLeg
    12: (243.0, 232.0, 88.0),  #leftForeArm
    13: (90., 64., 210.),      #rightForeArm
    14: (152.0, 200.0, 156.0), #leftUpLeg
    15: (129.0, 103.0, 106.0), #hips
}


class HumanSegmentationDataset():
    def __init__(self, file_list):
        self.file_list = file_list
        self.ORIG_BODY_PART_IDS = set(range(100, 126))

    def __len__(self):
        return len(self.file_list)

    def read_plyfile(self, file_path):
        """Read ply file and return it as numpy array. Returns None if emtpy."""
        with open(file_path, 'rb') as f:
            plydata = PlyData.read(f)
        if plydata.elements:
            return pd.DataFrame(plydata.elements[0].data).values

    def load_pc(self, file_path):
        pc = self.read_plyfile(file_path)  # (num_points, 8)

        pc_coords = pc[:, 0:3]  # (num_points, 3)
        pc_rgb = pc[:, 3:6].astype(np.uint8)  # (num_points, 3) - 0-255
        pc_orig_segm_labels = pc[:, 6].astype(np.uint8)  # (num_points,)
        return pc_coords, pc_rgb, torch.tensor(pc_orig_segm_labels).cuda()

    def export_colored_pcd_inst_segm(self, coords, pc_inst_labels, write_path):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        inst_colors = np.asarray([self.COLOR_MAP_INSTANCES[int(label_idx)] for label_idx in pc_inst_labels]) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(inst_colors)
        pcd.estimate_normals()
        o3d.io.write_point_cloud(write_path, pcd)

    def export_colored_pcd_part_segm(self, coords, pc_part_segm_labels, write_path):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        part_colors = np.asarray(
            [self.MERGED_BODY_PART_COLORS[int(label_idx)] for label_idx in pc_part_segm_labels]) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(part_colors)
        pcd.estimate_normals()
        o3d.io.write_point_cloud(write_path, pcd)

    def __getitem__(self, index):
        return self.load_pc(self.file_list[index])