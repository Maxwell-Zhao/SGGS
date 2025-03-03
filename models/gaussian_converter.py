import torch
import torch.nn as nn
import numpy as np
from .deformer import get_deformer
from .texture import get_texture
from models.unet.unet import Unet

class GaussianConverter(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata

        self.deformer = get_deformer(cfg.model.deformer, metadata)
        self.texture = get_texture(cfg.model.texture, metadata)

        self.delay = cfg.model.unet.get('delay', 0)
        self.unet = Unet(cfg.model.deformer.non_rigid, metadata, self.delay)

        self.optimizer, self.scheduler = None, None
        self.set_optimizer()

    def set_optimizer(self):
        opt_params = [
            {'params': self.deformer.rigid.parameters(), 'lr': self.cfg.opt.get('rigid_lr', 0.)},
            {'params': [p for n, p in self.deformer.non_rigid.named_parameters() if 'latent' not in n],
             'lr': self.cfg.opt.get('non_rigid_lr', 0.)},
            {'params': [p for n, p in self.deformer.non_rigid.named_parameters() if 'latent' in n],
             'lr': self.cfg.opt.get('nr_latent_lr', 0.), 'weight_decay': self.cfg.opt.get('latent_weight_decay', 0.05)},
            {'params': [p for n, p in self.texture.named_parameters() if 'latent' not in n],
             'lr': self.cfg.opt.get('texture_lr', 0.)},
            {'params': [p for n, p in self.texture.named_parameters() if 'latent' in n],
             'lr': self.cfg.opt.get('tex_latent_lr', 0.), 'weight_decay': self.cfg.opt.get('latent_weight_decay', 0.05)},
            {'params': [p for n, p in self.unet.named_parameters() if 'unet' in n],
             'lr': self.cfg.opt.get('unet_feature_lr', 0.)},
            {'params': [p for n, p in self.unet.named_parameters() if 'unet' not in n],
             'lr': self.cfg.opt.get('unet_lr', 0.)},
        ]
        self.optimizer = torch.optim.Adam(params=opt_params, lr=0.001, eps=1e-15)

        gamma = self.cfg.opt.lr_ratio ** (1. / self.cfg.opt.iterations)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

    def forward(self, gaussians, camera, iteration, compute_loss=True):
        loss_reg = {}

        deformed_gaussians, loss_reg_deformer = self.deformer(gaussians, camera, iteration, compute_loss)
        loss_reg.update(loss_reg_deformer)
        color_precompute = self.texture(deformed_gaussians, camera)

        if iteration >= self.delay:
            deformed_gaussians = self.unet.geometry_refine(deformed_gaussians, camera)

        return deformed_gaussians, loss_reg, color_precompute

    def optimize(self):
        grad_clip = self.cfg.opt.get('grad_clip', 0.)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()