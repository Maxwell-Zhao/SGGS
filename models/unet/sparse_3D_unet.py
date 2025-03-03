import torch
import torch.nn as nn
import sparseconvnet as scn
import torch
import torch.nn as nn
from models.network_utils import FusionMLP, HierarchicalPoseEncoder, HashGrid

# Hyperparameters
m = 8
residual_blocks = False  # True or False
block_reps = 1  # Conv block repetition factor: 1 or 2

# class sparse_3D_unet(nn.Module):
#     def __init__(self):
#         super(sparse_3D_unet, self).__init__()
#         self.sparseModel = scn.Sequential().add(
#             scn.InputLayer(3, 4096, mode=4)).add(
#             scn.SubmanifoldConvolution(3, 3, m, 3, False)).add(
#             scn.UNet(3, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
#             scn.BatchNormReLU(m)).add(
#             scn.OutputLayer(3))
#         self.linear = nn.Linear(m, 3)
#
#     def forward(self, x):
#         x = self.sparseModel(x)
#         x = self.linear(x)
#         return x


class sparse_3D_unet(nn.Module):
    def __init__(self):
        super(sparse_3D_unet, self).__init__()
        m = 4
        residual_blocks = True
        block_reps = 1

        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(3, 3072, mode=4)).add(
            scn.SubmanifoldConvolution(3, 3, m, 3, False)).add(
            scn.UNet(3, block_reps, [m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m], residual_blocks)).add(
            scn.BatchNormReLU(m)).add(
            scn.OutputLayer(3))
        self.linear = nn.Linear(m, 48)

    def forward(self, x):
        x = self.sparseModel(x)
        x = self.linear(x)
        return x