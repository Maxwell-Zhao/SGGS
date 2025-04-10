#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from scipy.spatial.transform import Rotation as R

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    resized_mask = PILtoTorch(cam_info.mask, resolution)
    gt_mask = resized_mask[:1, ...] != 0

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, frame_id=cam_info.frame_id, cam_id=cam_info.cam_id,
                  R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, mask=gt_mask, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  rots=cam_info.rots, Jtrs=cam_info.Jtrs, bone_transforms=cam_info.bone_transforms)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

# compute image space normal
def get_homo_2d(height, width):
    Y, X = np.meshgrid(np.arange(height, dtype=np.float32),
                       np.arange(width, dtype=np.float32),
                       indexing='ij')
    xy = np.stack([X, Y], axis=-1)
    homo_ones = np.ones((height, width, 1), dtype=np.float32)
    homo_2d = np.concatenate((xy, homo_ones), axis=2)
    return homo_2d

def get_inverse_intrinsic(K):
    K_inv = K.copy()
    K_inv[0, 0] = 1. / K[0, 0]
    K_inv[1, 1] = 1. / K[1, 1]
    K_inv[0, 2] = -K[0, 2] / K[0, 0]
    K_inv[1, 2] = -K[1, 2] / K[0, 0]
    return K_inv

def compute_normal_image(depth, fg_mask, camera):
    depth = depth.permute(1, 2, 0)

    height = camera.image_height
    width = camera.image_width

    homo_2d = get_homo_2d(height, width)
    K = camera.K
    K_inv = get_inverse_intrinsic(K)
    uv = np.dot(homo_2d.reshape([-1, 3]), K_inv.T).reshape([height, width, 3])
    uv = torch.from_numpy(uv).to(depth.device)
    cam_ray_dir = F.normalize(uv, p=2, dim=-1)

    pred_points = cam_ray_dir * depth

    zs = pred_points[:, :, 2]
    xs = pred_points[:, :, 0]
    ys = pred_points[:, :, 1]
    eps = 1e-10

    zy = (zs[1:, :] - zs[:-1, :]) / (ys[1:, :] - ys[:-1, :] + eps)
    zx = (zs[:, 1:] - zs[:, :-1]) / (xs[:, 1:] - xs[:, :-1] + eps)

    ny = torch.cat([-zy, torch.zeros(1, width, device=zy.device)], dim=0)
    nx = torch.cat([-zx, torch.zeros(height, 1, device=zx.device)], dim=1)
    nz = torch.ones(height, width, device=depth.device, dtype=torch.float32)
    pred_normals = torch.stack([nx, ny, nz], dim=0)

    n = torch.linalg.norm(pred_normals, dim=0, keepdim=True)
    pred_normals = pred_normals / n

    pred_normals[:, ~fg_mask] = -1
    pred_normals = ((pred_normals + 1) / 2.0).clip(0.0, 1.0)

    return pred_normals


def _update_extrinsics(
        extrinsics,
        angle,
        trans=None,
        rotate_axis='y'):
    r""" Uptate camera extrinsics when rotating it around a standard axis.

    Args:
        - extrinsics: Array (3, 3)
        - angle: Float
        - trans: Array (3, )
        - rotate_axis: String

    Returns:
        - Array (3, 3)
    """
    E = extrinsics
    inv_E = np.linalg.inv(E)

    camrot = inv_E[:3, :3]
    campos = inv_E[:3, 3]
    if trans is not None:
        campos -= trans

    rot_y_axis = camrot.T[1, 1]
    if rot_y_axis < 0.:
        angle = -angle

    rotate_coord = {
        'x': 0, 'y': 1, 'z': 2
    }
    grot_vec = np.array([0., 0., 0.])
    grot_vec[rotate_coord[rotate_axis]] = angle
    grot_mtx = cv2.Rodrigues(grot_vec)[0].astype('float32')

    rot_campos = grot_mtx.dot(campos)
    rot_camrot = grot_mtx.dot(camrot)
    if trans is not None:
        rot_campos += trans

    new_E = np.identity(4)
    new_E[:3, :3] = rot_camrot.T
    new_E[:3, 3] = -rot_camrot.T.dot(rot_campos)

    return new_E

def freeview_camera(camera, trans,
                    total_frames=100,
                    rotate_axis='z',
                    inv_angle=False,):

    cam_names = [str(cam_name) for cam_name in range(total_frames + 1)]
    all_cam_params = {'all_cam_names': cam_names}
    for frame_idx, cam_name in enumerate(cam_names):
        Ri = np.array(camera['R'], np.float32)
        Ti = np.array(camera['T'], np.float32)
        Ei = np.eye(4)
        Ei[:3,:3] = Ri
        Ei[:3,3:] = Ti

        angle = 2 * np.pi * (frame_idx / total_frames)
        if inv_angle:
            angle = -angle
        Eo = _update_extrinsics(Ei, angle, trans, rotate_axis)

        Ro = Eo[:3,:3]
        To = Eo[:3,3:]
        cam_params = {
            'K': camera['K'],
            'D': camera['D'],
            'R': Ro,
            'T': To,
        }

        all_cam_params.update({cam_name: cam_params})
    return all_cam_params

def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)

def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))

def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)

def look_at(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    if not opengl:
        # camera forward aligns with -z
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R

def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T

class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, near=0.01, far=100):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = np.deg2rad(fovy)  # deg 2 rad
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_matrix(np.eye(3))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!

    @property
    def fovx(self):
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        return self.pose[:3, 3]

    # pose (c2w)
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view (w2c)
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # projection (perspective)
    @property
    def perspective(self):
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def mvp(self):
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])