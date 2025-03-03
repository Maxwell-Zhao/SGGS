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

import os
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, neighborhood_consistency_loss
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import fix_random, Evaluator, PSEvaluator
from tqdm import tqdm
from utils.loss_utils import full_aiap_loss
from utils.dataset_utils import HumanSegmentationDataset
import hydra
from omegaconf import OmegaConf
import wandb
import lpips
from utils.loss_utils import adaptive_clustering


def C(iteration, value):
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = OmegaConf.to_container(value)
        if not isinstance(value, list):
            raise TypeError('Scalar specification only supports list, got', type(value))
        value_list = [0] + value
        i = 0
        current_step = iteration
        while i < len(value_list):
            if current_step >= value_list[i]:
                i += 2
            else:
                break
        value = value_list[i - 1]
    return value


def training(config):
    model = config.model
    dataset = config.dataset
    opt = config.opt
    pipe = config.pipeline
    testing_iterations = config.test_iterations
    testing_interval = config.test_interval
    saving_iterations = config.save_iterations
    checkpoint_iterations = config.checkpoint_iterations
    checkpoint = config.start_checkpoint
    debug_from = config.debug_from

    # define lpips
    lpips_type = config.opt.get('lpips_type', 'vgg')
    loss_fn_vgg = lpips.LPIPS(net=lpips_type).cuda()  # for training
    evaluator = PSEvaluator() if dataset.name == 'people_snapshot' else Evaluator()

    first_iter = 0
    gaussians = GaussianModel(model.gaussian)
    scene = Scene(config, gaussians, config.exp_dir)
    scene.train()

    bce_loss = nn.BCEWithLogitsLoss()

    gaussians.training_setup(opt)
    if checkpoint:
        scene.load_checkpoint(checkpoint)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    data_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # =====================================================================================================================
    # get semantic label
    pcd_path = './body_models/smpl/neutral/smpl_semantic.ply'  # sample input scene
    file_list = [pcd_path]  # for now just the demo scene
    pre_dataset = HumanSegmentationDataset(file_list=file_list)
    coords, colors, labels = pre_dataset.load_pc(pcd_path)

    gaussians.frozen_labels = labels.cuda()
    gaussians._objects_dc = F.one_hot(gaussians.frozen_labels.to(torch.int64), num_classes=20).unsqueeze(1).to(
        torch.float32)

    # =====================================================================================================================

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random data point
        if not data_stack:
            data_stack = list(range(len(scene.train_dataset)))
        data_idx = data_stack.pop(randint(0, len(data_stack) - 1))
        data = scene.train_dataset[data_idx]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        ####################################################################
        use_semantic = True
        densify_grad_threshold = 0.001
        if use_semantic == True and iteration in [5000, 10000]:
            frozen_labels = torch.argmax(gaussians._objects_dc.squeeze(1), dim=1)
            features = torch.cat((gaussians._features_dc.squeeze(1), gaussians._opacity), dim=1)
            grads = torch.zeros(gaussians._xyz.shape[0], device=features.device)
            # head, torso, hips
            for i in [4, 7, 15]:
                result = adaptive_clustering(features, gaussians._xyz, frozen_labels, i)
                grads[int(result)] = opt.densify_grad_threshold

            # 自适应密集化
            adaptive_threshold = densify_grad_threshold * (1 + torch.mean(grads))
            gaussians.densify_and_clone(grads, adaptive_threshold, scene.cameras_extent)
        ####################################################################

        lambda_mask = C(iteration, config.opt.lambda_mask)
        use_mask = lambda_mask > 0.
        render_pkg = render(data, iteration, scene, pipe, background, compute_loss=True, return_opacity=use_mask,
                            semantic=True)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]
        opacity = render_pkg["opacity_render"] if use_mask else None
        rendered_object = render_pkg["rendered_object"]
        deformed_gaussian = render_pkg["deformed_gaussian"]

        # Loss
        gt_image = data.original_image.cuda()

        lambda_l1 = C(iteration, config.opt.lambda_l1)
        lambda_dssim = C(iteration, config.opt.lambda_dssim)
        loss_l1 = torch.tensor(0.).cuda()
        loss_dssim = torch.tensor(0.).cuda()
        if lambda_l1 > 0.:
            loss_l1 = l1_loss(image, gt_image)
        if lambda_dssim > 0.:
            loss_dssim = 1.0 - ssim(image, gt_image)
        loss = lambda_l1 * loss_l1 + lambda_dssim * loss_dssim

        # perceptual loss
        lambda_perceptual = C(iteration, float(config.opt.get('lambda_perceptual', 0.)))
        if lambda_perceptual > 0:
            # crop the foreground
            mask = data.original_mask.cpu().numpy()
            mask = np.where(mask)
            y1, y2 = mask[1].min(), mask[1].max() + 1
            x1, x2 = mask[2].min(), mask[2].max() + 1
            fg_image = image[:, y1:y2, x1:x2]
            gt_fg_image = gt_image[:, y1:y2, x1:x2]

            loss_perceptual = loss_fn_vgg(fg_image, gt_fg_image, normalize=True).mean()
            loss += lambda_perceptual * loss_perceptual
        else:
            loss_perceptual = torch.tensor(0.)

        # mask loss
        gt_mask = data.original_mask.cuda()
        if not use_mask:
            loss_mask = torch.tensor(0.).cuda()
        elif config.opt.mask_loss_type == 'bce':
            opacity = torch.clamp(opacity, 1.e-3, 1. - 1.e-3)
            loss_mask = F.binary_cross_entropy(opacity, gt_mask)
        elif config.opt.mask_loss_type == 'l1':
            loss_mask = F.l1_loss(opacity, gt_mask)
        else:
            raise ValueError
        loss += lambda_mask * loss_mask

        # skinning loss
        lambda_skinning = C(iteration, config.opt.lambda_skinning)
        if lambda_skinning > 0:
            loss_skinning = scene.get_skinning_loss()
            loss += lambda_skinning * loss_skinning
        else:
            loss_skinning = torch.tensor(0.).cuda()

        lambda_aiap_xyz = C(iteration, config.opt.get('lambda_aiap_xyz', 0.))
        lambda_aiap_cov = C(iteration, config.opt.get('lambda_aiap_cov', 0.))
        if lambda_aiap_xyz > 0. or lambda_aiap_cov > 0.:
            loss_aiap_xyz, loss_aiap_cov = full_aiap_loss(scene.gaussians, render_pkg["deformed_gaussian"])
        else:
            loss_aiap_xyz = torch.tensor(0.).cuda()
            loss_aiap_cov = torch.tensor(0.).cuda()
        loss += lambda_aiap_xyz * loss_aiap_xyz
        loss += lambda_aiap_cov * loss_aiap_cov

        # regularization
        loss_reg = render_pkg["loss_reg"]
        for name, value in loss_reg.items():
            lbd = opt.get(f"lambda_{name}", 0.)
            lbd = C(iteration, lbd)
            loss += lbd * value

        ####################################################################
        # semantic
        # if iteration > opt.coarse_iterations:
        #     lambda_semantic = C(iteration, float(config.opt.get('lambda_semantic', 0.)))
        #     lambda_neighborhood = C(iteration, float(config.opt.get('lambda_neighborhood', 0.)))
        #
        #     loss_consistency = neighborhood_consistency_loss(gaussians._xyz.squeeze().detach(), gaussians._objects_dc.squeeze(1).detach())
        #     loss += loss_consistency * lambda_neighborhood
        #
        #     semantic = data.semantic.cuda()
        #     object_type = torch.argmax(rendered_object.permute(1, 2, 0), dim=2)
        #     device = object_type.device
        #
        #     body_parts = {
        #         'spine': {7},
        #         'leg': {2, 5, 6, 8, 11, 14},
        #         'hand': {1, 3, 9, 10, 12, 13},
        #         'head': {4},
        #         'hips': {15}
        #     }
        #     body_part_masks = {part: torch.isin(object_type, torch.tensor(list(types), device=device))
        #                        for part, types in body_parts.items()}
        #
        #     spine = body_part_masks['spine']
        #     leg = body_part_masks['leg']
        #     hand = body_part_masks['hand']
        #     head = body_part_masks['head']
        #     hips = body_part_masks['hips']
        #
        #     spine_color = torch.tensor([226, 226, 226], device=device)
        #     leg_color = torch.tensor([129, 0, 50], device=device)
        #     hand_color = torch.tensor([243, 115, 68], device=device)
        #     head_color = torch.tensor([228, 162, 227], device=device)
        #     hips_color = torch.tensor([210, 78, 142], device=device)
        #
        #     gt_spine = torch.all(semantic == spine_color.view(3, 1, 1), dim=0,)
        #     gt_leg = torch.all(semantic == leg_color.view(3, 1, 1), dim=0,)
        #     gt_hand = torch.all(semantic == hand_color.view(3, 1, 1), dim=0,)
        #     gt_head = torch.all(semantic == head_color.view(3, 1, 1), dim=0,)
        #     gt_hips = torch.all(semantic == hips_color.view(3, 1, 1), dim=0,)
        #
        #     semantic_loss = (bce_loss(spine*1.,gt_spine*1.)+bce_loss(leg*1.,gt_leg*1.)+
        #                      bce_loss(hand*1.,gt_hand*1.)+bce_loss(head*1.,gt_head*1.)+bce_loss(hips*1.,gt_hips*1.))
        #
        #     loss += semantic_loss * lambda_semantic

        ####################################################################

        def labels_to_rgb(labels, device='cuda'):
            # 定义身体部位
            body_parts = {
                'spine': {7},
                'leg': {2, 5, 6, 8, 11, 14},
                'hand': {1, 3, 9, 10, 12, 13},
                'head': {4},
                'hips': {15}
            }

            # 定义颜色
            colors = {
                'spine': torch.tensor([226, 226, 226], device=device, dtype=torch.float),
                'leg': torch.tensor([129, 0, 50], device=device, dtype=torch.float),
                'hand': torch.tensor([243, 115, 68], device=device, dtype=torch.float),
                'head': torch.tensor([228, 162, 227], device=device, dtype=torch.float),
                'hips': torch.tensor([210, 78, 142], device=device, dtype=torch.float)
            }

            # 创建输出tensor
            n = labels.size(0)
            rgb = torch.zeros((n, 3), device=device)

            # 为每个标签分配颜色
            for part, label_set in body_parts.items():
                for label in label_set:
                    rgb[labels == label] = colors[part]

            return rgb

        save_ply = True
        if (save_ply) and (iteration % 1000 == 0):
            from utils.dataset_utils import storePly
            xyz = deformed_gaussian.get_xyz
            object = deformed_gaussian.get_objects
            frozen_labels = torch.argmax(object.squeeze(1), dim=1)
            rgb = labels_to_rgb(frozen_labels.float())
            ply_path = 'ply1/' + str(iteration) + '.ply'

            xyz = xyz.detach().cpu().numpy()
            rgb = rgb.detach().cpu().numpy()

            storePly(ply_path, xyz, rgb)

        loss.backward()
        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            elapsed = iter_start.elapsed_time(iter_end)
            log_loss = {
                'loss/l1_loss': loss_l1.item(),
                'loss/ssim_loss': loss_dssim.item(),
                'loss/perceptual_loss': loss_perceptual.item(),
                'loss/mask_loss': loss_mask.item(),
                'loss/loss_skinning': loss_skinning.item(),
                'loss/xyz_aiap_loss': loss_aiap_xyz.item(),
                'loss/cov_aiap_loss': loss_aiap_cov.item(),
                'loss/total_loss': loss.item(),
                'iter_time': elapsed,
            }
            log_loss.update({
                'loss/loss_' + k: v for k, v in loss_reg.items()
            })
            wandb.log(log_loss)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            validation(iteration, testing_iterations, testing_interval, scene, evaluator, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and iteration > model.gaussian.delay:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt, scene, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                scene.optimize(iteration)

            if iteration in checkpoint_iterations:
                scene.save_checkpoint(iteration)


def validation(iteration, testing_iterations, testing_interval, scene: Scene, evaluator, renderArgs):
    # Report test and samples of training set
    if testing_interval > 0:
        if not (iteration % testing_interval == 0 and iteration > 3000):
            return
    else:
        if not iteration in testing_iterations:
            return

    scene.eval()
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras': list(range(len(scene.test_dataset)))},
                          {'name': 'train', 'cameras': [idx for idx in range(0, len(scene.train_dataset),
                                                                             len(scene.train_dataset) // 10)]})

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            examples = []
            for idx, data_idx in enumerate(config['cameras']):
                data = getattr(scene, config['name'] + '_dataset')[data_idx]
                render_pkg = render(data, iteration, scene, *renderArgs, compute_loss=False, return_opacity=True)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(data.original_image.to("cuda"), 0.0, 1.0)
                opacity_image = torch.clamp(render_pkg["opacity_render"], 0.0, 1.0)

                wandb_img = wandb.Image(opacity_image[None],
                                        caption=config['name'] + "_view_{}/render_opacity".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(image[None], caption=config['name'] + "_view_{}/render".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(gt_image[None], caption=config['name'] + "_view_{}/ground_truth".format(
                    data.image_name))
                examples.append(wandb_img)

                l1_test += l1_loss(image, gt_image).mean().double()
                metrics_test = evaluator(image, gt_image)
                psnr_test += metrics_test["psnr"]
                ssim_test += metrics_test["ssim"]
                lpips_test += metrics_test["lpips"]

                wandb.log({config['name'] + "_images": examples})
                examples.clear()

            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test,
                                                                                     psnr_test, ssim_test, lpips_test))
            wandb.log({
                config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                config['name'] + '/loss_viewpoint - psnr': psnr_test,
                config['name'] + '/loss_viewpoint - ssim': ssim_test,
                config['name'] + '/loss_viewpoint - lpips': lpips_test,
            })

    wandb.log({'scene/opacity_histogram': wandb.Histogram(scene.gaussians.get_opacity.cpu())})
    wandb.log({'total_points': scene.gaussians.get_xyz.shape[0]})
    torch.cuda.empty_cache()
    scene.train()


def main(config):
    # print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False)  # allow adding new values to config

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)
    config.checkpoint_iterations.append(config.opt.iterations)

    # set wandb logger
    wandb_name = config.name
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project='gaussian-splatting-avatar',
        # entity='fast-avatar',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )#
    #print("Optimizing " + config.exp_dir)

    # Initialize system state (RNG)
    fix_random(config.seed)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(config.detect_anomaly)
    training(config)

    # All done
    print("\nTraining complete.")

import yaml

def load_config(config_path, config_name):
    with open(os.path.join(config_path, f"{config_name}.yaml"), "r") as f:
        config = yaml.safe_load(f)
    return OmegaConf.create(config)

if __name__ == "__main__":
    config = load_config("configs", "config_zju")
    main(config)

    # print("Optimizing " + config.exp_dir)

    # Initialize system state (RNG)
    fix_random(config.seed)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(config.detect_anomaly)
    training(config)

    # All done
    print("\nTraining complete.")


