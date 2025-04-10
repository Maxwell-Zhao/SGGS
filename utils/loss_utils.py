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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from sklearn.neighbors import KDTree
import numpy as np
import cv2
from pytorch3d.ops.knn import knn_points

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def full_aiap_loss(gs_can, gs_obs, n_neighbors=5):
    xyz_can = gs_can.get_xyz
    xyz_obs = gs_obs.get_xyz

    cov_can = gs_can.get_covariance()
    cov_obs = gs_obs.get_covariance()

    _, nn_ix, _ = knn_points(xyz_can.unsqueeze(0),
                             xyz_can.unsqueeze(0),
                             K=n_neighbors,
                             return_sorted=True)
    nn_ix = nn_ix.squeeze(0)

    loss_xyz = aiap_loss(xyz_can, xyz_obs, nn_ix=nn_ix)
    loss_cov = aiap_loss(cov_can, cov_obs, nn_ix=nn_ix)

    return loss_xyz, loss_cov

def aiap_loss(x_canonical, x_deformed, n_neighbors=5, nn_ix=None):
    if x_canonical.shape != x_deformed.shape:
        raise ValueError("Input point sets must have the same shape.")

    if nn_ix is None:
        _, nn_ix, _ = knn_points(x_canonical.unsqueeze(0),
                                 x_canonical.unsqueeze(0),
                                 K=n_neighbors + 1,
                                 return_sorted=True)
        nn_ix = nn_ix.squeeze(0)

    dists_canonical = torch.cdist(x_canonical.unsqueeze(1), x_canonical[nn_ix])[:,0,1:]
    dists_deformed = torch.cdist(x_deformed.unsqueeze(1), x_deformed[nn_ix])[:,0,1:]

    loss = F.l1_loss(dists_canonical, dists_deformed)

    return loss

def neighborhood_consistency_loss(features, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
    # Conditionally downsample if points exceed max_points
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]
        predictions = predictions[indices]

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices]
    sample_preds = predictions[indices]

    # print("sample_features={}".format(sample_features))
    # print("features={}".format(features))
    # print("sample_preds={}".format(sample_preds))
    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_features, features)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

    # print("dists={}".format(dists))
    # Fetch neighbor predictions using indexing
    neighbor_preds = predictions[neighbor_indices_tensor]
    # print("neighbor_preds={}".format(neighbor_preds))

    # Compute KL divergence
    # print("sample_preds={}".format(torch.log(sample_preds.unsqueeze(1) + 1e-6)))
    # print("neighbor_preds={}".format(neighbor_preds))
    sample_preds = torch.clamp(sample_preds, min=1e-7, max=1 - 1e-7)
    neighbor_preds = torch.clamp(neighbor_preds, min=1e-7, max=1 - 1e-7)
    kl = sample_preds.unsqueeze(1) * (torch.log(sample_preds.unsqueeze(1)) - torch.log(neighbor_preds))
    # print("kl={}".format(kl))
    loss = kl.sum(dim=-1).mean()

    # print("loss={}".format(loss))
    # Normalize loss into [0, 1]
    num_classes = predictions.size(1)
    normalized_loss = loss / num_classes

    return lambda_val * normalized_loss

def s3im_fun(src_vec, tar_vec,repeat_time=10):
    #     r"""Implements Stochastic Structural SIMilarity(S3IM) algorithm.
    #     It is proposed in the ICCV2023 paper
    #     `S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields`.
    index_list = []
    channel,patch_height,patch_width = src_vec.shape
    src_vec = src_vec.reshape(1,-1)
    tar_vec= tar_vec.reshape(1,-1)
    for i in range(repeat_time):
        if i == 0:
            tmp_index = torch.arange(len(tar_vec))
            index_list.append(tmp_index)
        else:
            ran_idx = torch.randperm(len(tar_vec))
            index_list.append(ran_idx)
    res_index = torch.cat(index_list)
    tar_all = tar_vec[res_index]
    src_all = src_vec[res_index]
    tar_patch = tar_all.permute(1, 0).reshape(1, channel, patch_height, patch_width * repeat_time)
    src_patch = src_all.permute(1, 0).reshape(1, channel, patch_height, patch_width * repeat_time)
    loss = (1 - ssim(src_patch, tar_patch))
    return loss


def point_to_graph(point_cloud_distance, point_cloud_attr, k):
    kdtree = KDTree(point_cloud_distance, leaf_size=30, metric='euclidean')
    distances, indices = kdtree.query(point_cloud_distance, k)
    edge_index = []
    edge_attr = []
    for i in range(len(point_cloud_distance)):
        for index, j in enumerate(indices[i]):
            if i != j:
                edge_index.append([i, j])
                edge_attr.append(distances[i, index])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    data = Data(x=torch.tensor(point_cloud_attr, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
    return data

from scipy.spatial import cKDTree
def point_to_graph(point_cloud_distance, point_cloud_attr, k):
    kdtree = cKDTree(point_cloud_distance)
    distances, indices = kdtree.query(point_cloud_distance, k=k + 1)  # k+1 to include self

    edge_index = []
    edge_attr = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        for d, j in zip(dist[1:], idx[1:]):  # Skip first (self)
            edge_index.append([i, j])
            edge_attr.append(d)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

    data = Data(x=torch.tensor(point_cloud_attr, dtype=torch.float),
                edge_index=edge_index,
                edge_attr=edge_attr)
    return data

import math
def adaptive_clustering(features, pos, frozen_labels, label):
    result, max_score = None, None
    origin_id = torch.arange(features.shape[0], device=features.device)
    part_feature = features[frozen_labels == label]
    part_pos = pos[frozen_labels == label]
    origin_id = origin_id[frozen_labels == label]

    graph = to_networkx(point_to_graph(part_pos.detach().cpu().numpy(), part_feature.detach().cpu().numpy(), 3),
                        to_undirected=True)

    for i in graph.nodes:
        node_score = 0
        for j in graph.neighbors(i):
            r1, g1, b1, a1 = part_feature[i][:4]
            r2, g2, b2, a2 = part_feature[j][:4]

            color_score = torch.sqrt((r2 - r1) ** 2 + (g2 - g1) ** 2 + (b2 - b1) ** 2) / math.sqrt(3)
            opacity_score = torch.abs(a1 - a2)
            dist_score = torch.norm(part_pos[i] - part_pos[j])
            score = color_score + opacity_score + 0.01 * dist_score
            node_score += score

        if max_score is None or node_score > max_score:
            max_score = node_score
            result = origin_id[i]

    return result.item()


# def adaptive_clustering(features, pos, frozen_labels, label):
#     device = features.device
#     part_mask = frozen_labels == label
#     part_feature = features[part_mask]
#     part_pos = pos[part_mask]
#     origin_id = torch.arange(features.shape[0], device=device)[part_mask]
#
#     graph = to_networkx(point_to_graph(part_pos.detach().cpu().numpy(), part_feature.detach().cpu().numpy(), k=6),
#                         to_undirected=True)
#
#     # 计算全局特征
#     mean_color = part_feature[:, :3].mean(dim=0)
#     mean_opacity = part_feature[:, 3].mean()
#     center_pos = part_pos.mean(dim=0)
#
#     max_score = None
#     result = None
#
#     for i in graph.nodes:
#         node_score = 0
#         local_feature = part_feature[i]
#         local_pos = part_pos[i]
#
#         for j in graph.neighbors(i):
#             neighbor_feature = part_feature[j]
#             neighbor_pos = part_pos[j]
#
#             # 计算局部特征差异
#             color_diff = torch.norm(local_feature[:3] - neighbor_feature[:3])
#             opacity_diff = abs(local_feature[3] - neighbor_feature[3])
#             pos_diff = torch.norm(local_pos - neighbor_pos)
#
#             # 局部得分
#             local_score = color_diff + opacity_diff + 0.01 * pos_diff
#             node_score += local_score
#
#         # 计算全局特征差异
#         global_color_diff = torch.norm(local_feature[:3] - mean_color)
#         global_opacity_diff = abs(local_feature[3] - mean_opacity)
#         global_pos_diff = torch.norm(local_pos - center_pos)
#
#         # 全局得分
#         global_score = 0.1 * global_color_diff + 0.1 * global_opacity_diff + 0.005 * global_pos_diff
#
#         # 综合得分
#         total_score = node_score + global_score
#
#         if max_score is None or total_score > max_score:
#             max_score = total_score
#             result = origin_id[i]
#
#     return result.item()
