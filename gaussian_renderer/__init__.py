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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import diff_gaussian_rasterization_obj as dgro

def render(data,
           iteration,
           scene,
           pipe,
           bg_color : torch.Tensor,
           scaling_modifier = 1.0,
           override_color = None,
           compute_loss=True,
           return_opacity=False,
           semantic=False,):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    pc, loss_reg, colors_precomp = scene.convert_gaussians(data, iteration, compute_loss)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(data.FoVx * 0.5)
    tanfovy = math.tan(data.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data.image_height),
        image_width=int(data.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data.world_view_transform,
        projmatrix=data.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=data.camera_center,
        prefiltered=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if semantic:
        raster_settings_obj = dgro.GaussianRasterizationSettings(
            image_height=int(data.image_height),
            image_width=int(data.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=data.world_view_transform,
            projmatrix=data.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=data.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer_obj = dgro.GaussianRasterizer(raster_settings=raster_settings_obj)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    opacity_image = None
    rendered_object = None

    if semantic:
        labels = torch.argmax(pc.get_objects.squeeze(1), dim=1)
        device = 'cuda'
        n = labels.size(0)

        rgb_spine = torch.zeros((n, 3), device=device)
        rgb_spine[labels==7] = torch.tensor([1.0, 1.0, 1.0], device=device)

        rgb_leg = torch.zeros((n, 3), device=device)
        mask_leg = (labels==2) | (labels==5) | (labels==6) | (labels==8) | (labels==11) | (labels==14)
        rgb_leg[mask_leg] = torch.tensor([1.0, 1.0, 1.0], device=device)

        rgb_hand = torch.zeros((n, 3), device=device)
        mask_hand = (labels==1) | (labels==3) | (labels==9) | (labels==10) | (labels==12) | (labels==13)
        rgb_hand[mask_hand] = torch.tensor([1.0, 1.0, 1.0], device=device)

        rgb_head = torch.zeros((n, 3), device=device)
        rgb_head[labels == 4] = torch.tensor([1.0, 1.0, 1.0], device=device)

        rgb_hips = torch.zeros((n, 3), device=device)
        rgb_hips[labels==15] = torch.tensor([1.0, 1.0, 1.0], device=device)

        semantic_opacity = torch.ones_like(opacity)

        rendered_spine, _, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=rgb_spine,
            opacities=semantic_opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        render_leg,_,_ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=rgb_leg,
            opacities=semantic_opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        render_hand,_,_ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=rgb_hand,
            opacities=semantic_opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        render_head,_,_ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=rgb_head,
            opacities=semantic_opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        render_hips,_,_ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=rgb_hips,
            opacities=semantic_opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        rendered_object = {
            "spine": rendered_spine,
            "leg": render_leg,
            "hand": render_hand,
            "head": render_head,
            "hips": render_hips
        }


    if return_opacity:
        opacity_image, _, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=torch.ones(opacity.shape[0], 3, device=opacity.device),
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        opacity_image = opacity_image[:1]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"deformed_gaussian": pc,
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "loss_reg": loss_reg,
            "opacity_render": opacity_image,
            "rendered_object": rendered_object,
            }
