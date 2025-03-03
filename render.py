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
import numpy as np
from scene import Scene
import os
from tqdm import tqdm, trange
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import fix_random
from scene import GaussianModel

from utils.general_utils import Evaluator, PSEvaluator

import hydra
from omegaconf import OmegaConf
import wandb
import time

def predict(config):
    with torch.set_grad_enabled(False):
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        times = []
        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]
            iter_start.record()

            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background,
                                compute_loss=False, return_opacity=False)
            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["render"]

            wandb_img = [wandb.Image(rendering[None], caption='render_{}'.format(view.image_name)),]
            wandb.log({'test_images': wandb_img})

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{view.image_name}.png"))

            # evaluate
            times.append(elapsed)

        _time = np.mean(times[1:])
        wandb.log({'metrics/time': _time})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'),
                 time=_time)

def test(config):
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        evaluator = PSEvaluator() if config.dataset.name == 'people_snapshot' else Evaluator()

        psnrs = []
        fpss = []
        ssims = []
        lpipss = []
        times = []
        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]
            iter_start.record()

            frame_start_time = time.time()
            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background,
                                compute_loss=False, return_opacity=False)
            frame_end_time = time.time()


            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["render"]

            gt = view.original_image[:3, :, :]

            wandb_img = [wandb.Image(rendering[None], caption='render_{}'.format(view.image_name)),
                         wandb.Image(gt[None], caption='gt_{}'.format(view.image_name))]

            wandb.log({'test_images': wandb_img})

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{view.image_name}.png"))

            # evaluate
            if config.evaluate:
                frame_time = frame_end_time - frame_start_time
                fps = 1.0 / frame_time
                metrics = evaluator(rendering, gt)
                psnrs.append(metrics['psnr'])
                ssims.append(metrics['ssim'])
                lpipss.append(metrics['lpips'])
                fpss.append(fps)
            else:
                psnrs.append(torch.tensor([0.], device='cuda'))
                ssims.append(torch.tensor([0.], device='cuda'))
                lpipss.append(torch.tensor([0.], device='cuda'))
                fpss.append(torch.tensor([0.], device='cuda'))
            times.append(elapsed)

        _psnr = torch.mean(torch.stack(psnrs))
        _ssim = torch.mean(torch.stack(ssims))
        _lpips = torch.mean(torch.stack(lpipss))
        _time = np.mean(times[1:])
        _fps = np.mean(fpss)
        print('psnr',_psnr)
        print('ssim',_ssim)
        print('lpips',_lpips)
        print('fps',_fps)
        wandb.log({'metrics/psnr': _psnr,
                   'metrics/ssim': _ssim,
                   'metrics/lpips': _lpips,
                   'metrics/time': _time,
                   'metrics/fps': _fps})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'),
                 psnr=_psnr.cpu().numpy(),
                 ssim=_ssim.cpu().numpy(),
                 lpips=_lpips.cpu().numpy(),
                 time=_time)

def main(config):
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)

    # set wandb logger
    if config.mode == 'test':
        config.suffix = config.mode + '-' + config.dataset.test_mode
    elif config.mode == 'predict':
        predict_seq = config.dataset.predict_seq
        if config.dataset.name == 'zjumocap':
            predict_dict = {
                0: 'dance0',
                1: 'dance1',
                2: 'flipping',
                3: 'canonical'
            }
        else:
            predict_dict = {
                0: 'rotation',
                1: 'dance2',
            }
        predict_mode = predict_dict[predict_seq]
        config.suffix = config.mode + '-' + predict_mode
    else:
        raise ValueError
    if config.dataset.freeview:
        config.suffix = config.suffix + '-freeview'
    wandb_name = config.name + '-' + config.suffix
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project='gaussian-splatting-avatar-test',
        # entity='fast-avatar',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )

    fix_random(config.seed)

    if config.mode == 'test':
        test(config)
    elif config.mode == 'predict':
        predict(config)
    else:
        raise ValueError

import yaml
def load_config(config_path, config_name):
    with open(os.path.join(config_path, f"{config_name}.yaml"), "r") as f:
        config = yaml.safe_load(f)
    return OmegaConf.create(config)

if __name__ == "__main__":
    config = load_config("configs", "config_zju")
    config.mode = 'test'
    main(config)