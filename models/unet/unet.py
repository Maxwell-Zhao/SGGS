import torch
import torch.nn as nn
import sparseconvnet as scn
import torch
import torch.nn as nn
from models.network_utils import FusionMLP, HierarchicalPoseEncoder, HashGrid
from models.unet.sparse_3D_unet import sparse_3D_unet


class Unet(nn.Module):
    def __init__(self,cfg,metadata, device=None, delay=0):
        super().__init__()
        # self.geometry_unet = sparse_3D_unet()
        self.device = device
        self.delay = delay

        self.pose_encoder = HierarchicalPoseEncoder(**cfg.pose_encoder)
        d_cond = self.pose_encoder.n_output_dims
        d_out = 3 + 3 + 4
        self.hashgrid_xyz = HashGrid(cfg.hashgrid)
        self.hashgrid_gemetry = HashGrid(cfg.hashgrid)
        self.aabb = metadata['aabb']

        self.fusionMLP = FusionMLP(48, 144, d_out, cfg.mlp)
        self.scale_factor = nn.Parameter(torch.tensor([0.001]))

        m = 8
        residual_blocks = True
        block_reps = 1
        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(3, 3072, mode=4)).add(
            scn.SubmanifoldConvolution(3, 3, m, 3, False)).add(
            scn.UNet(3, block_reps, [m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m], residual_blocks)).add(
            scn.BatchNormReLU(m)).add(
            scn.OutputLayer(3))
        self.linear = nn.Linear(m, 48)

    def geometry_unet(self, x):
        x = self.sparseModel(x)
        x = self.linear(x)
        return x

    def geometry_refine(self,deformaed_gaussians,camera):

        rots = camera.rots
        Jtrs = camera.Jtrs
        pose_feat = self.pose_encoder(rots, Jtrs)

        xyz = deformaed_gaussians.get_xyz
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        feature_xyz = self.hashgrid_xyz(xyz_norm)

        min_coord = -2.0
        voxel_size = 0.001
        feature_geometry = ((xyz - min_coord) / voxel_size).floor().long()

        # frozen_labels = torch.argmax(deformaed_gaussians._objects_dc.squeeze(1), dim=1)
        # n = deformaed_gaussians._objects_dc.shape[0]
        # feature_geometry_feature = torch.zeros((n, 3), dtype=torch.float32).cuda()
        #
        # feature_geometry_feature[:, 0] = (frozen_labels // 1) % 5
        # feature_geometry_feature[:, 1] = (frozen_labels // 5) % 4
        # feature_geometry_feature[:, 2] = (frozen_labels // 20) % 3

        # geometry_infor = [feature_geometry,feature_geometry_feature]
        geometry_infor = [feature_geometry,feature_geometry.float()]

        feature_geometry = self.geometry_unet(geometry_infor)

        deltas = self.fusionMLP(feature_xyz, feature_geometry, cond=pose_feat)

        delta_xyz = deltas[:, :3]*self.scale_factor
        delta_scale = deltas[:, 3:6]*self.scale_factor
        delta_rot = deltas[:, 6:10]*self.scale_factor

        deformaed_gaussians._xyz = deformaed_gaussians._xyz + delta_xyz
        deformaed_gaussians._scaling = deformaed_gaussians._scaling + delta_scale
        deformaed_gaussians._rotation = deformaed_gaussians._rotation + delta_rot

        return deformaed_gaussians




