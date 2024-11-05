import torch
import os
import wandb 
import torchvision.transforms as tr
import numpy as np
import sys
import random
import string
import tqdm 
import math
import matplotlib.pyplot as plt
from functools import partial
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from ae.utils import make_dashboard_video
from ae.utils import AverageMeter
from PIL import Image
from ae.normalizer import renormalize_img



to_pil_transform = tr.Compose([
    tr.Normalize(-1, 2),
    tr.Lambda(lambda x : x.clamp(0, 1)),
    tr.ToPILImage()
])

def make_injection_dashboard(epoch, imgs, n_imgs): 
    #TODO: Check this function
    n_latent_channel = imgs['latent_imgs'].size(1)
    n_rows = 2 + n_latent_channel
    n_cols = 2 * n_imgs
    figsize_x = 1.5 * n_cols + 0.5
    figsize_y = 1.5 * n_rows
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(figsize_x, figsize_y),
        constrained_layout=True
    )

    fig.suptitle(f"Epoch {epoch}")

    for k in range(2):
        for i in range(0, n_imgs):
            x_pad = 0 if k == 0 else n_imgs
            axes[0, i + x_pad].imshow(
                to_pil_transform(imgs['gt_imgs'][i])
            )
            #axes[0, i + x_pad].set_title(f"GT")
            axes[0, i + x_pad].axis('off')

            latent_img = renormalize_img(imgs['latent_imgs'][i]).permute(1, 2, 0) # [C, H, W] -> [H, W, C]
            for j in range(n_latent_channel):

                if k == 0 :
                    unreg_plot = axes[j+1, i + x_pad].imshow(
                        latent_img[..., j].detach().cpu(), 
                        vmin=0.0, vmax=1.0
                    )
                elif k ==1 : 
                    reg_plot = axes[j+1, i + x_pad].imshow(
                        latent_img[..., j].detach().cpu()
                    )

                #axes[j+1, i + x_pad].set_title(f"c{j}")
                axes[j+1, i + x_pad].axis('off')

            axes[j+2, i + x_pad].imshow(
                to_pil_transform(imgs['recon_imgs'][i])
            )
            #axes[j+2, i + x_pad].set_title(f"Recon")
            axes[j+2, i + x_pad].axis('off')

    cb1 = fig.colorbar(unreg_plot, ax=axes[1:n_latent_channel+1, 0::2], location='left', shrink=0.5)
    cb2 = fig.colorbar(reg_plot, ax=axes[1:n_latent_channel+1, 1::2], location='right', shrink=0.5)

    return fig

def make_dashboard(epoch:int, views:dict, n_views:int):

    num_channels_pred = views['latent_img_render'].size(1)
    num_channel_encoded = views['latent_img'].size(1)
    
    fig, axes = plt.subplots(
        nrows= 2* n_views, 
        ncols=max(num_channels_pred, num_channel_encoded) + 2, 
        figsize=(2*max(num_channels_pred, num_channel_encoded), 4* n_views)
    )
    fig.suptitle(f"Epoch {epoch}")

    for j in range(n_views): 
        render_img = renormalize_img(views['latent_img_render'][j]).permute(1, 2, 0)
        encoded_gt_img = renormalize_img(views['latent_img'][j]).permute(1, 2, 0)
        decoded_render_img = to_pil_transform(views['rgb_img_render'][j])
        gt_rgb = to_pil_transform(views['rgb_img'][j])
        decoded_encoded_gt_img = to_pil_transform(views['rgb_img_ae'][j])

        for i in range(num_channels_pred):
            axes[2*j + 0, i].imshow(render_img[..., i].detach().cpu(), vmin=0.0, vmax=1.0)
            axes[2*j + 0, i].set_title(f"Render c{i}")
            axes[2*j + 0, i].axis('off')
        axes[2*j + 0, i+1].imshow(decoded_render_img)
        axes[2*j + 0, i+1].set_title("Render dec")
        axes[2*j + 0, i+1].axis('off')
        axes[2*j + 0, i+2].axis('off')

        for i in range(num_channel_encoded):
            axes[2*j + 1, i].imshow(encoded_gt_img[..., i].detach().cpu(), vmin=0.0, vmax=1.0)
            axes[2*j + 1, i].set_title(f"GT enc c{i}")
            axes[2*j + 1, i].axis('off')

        axes[2*j + 1, i+1].imshow(gt_rgb)
        axes[2*j + 1, i+1].set_title("GT")
        axes[2*j + 1, i+1].axis('off')

        axes[2*j + 1, i+2].imshow(decoded_encoded_gt_img)
        axes[2*j + 1, i+2].set_title("AutoEnc")
        axes[2*j + 1, i+2].axis('off')

    return fig

class IdxSampledDataset(torch.utils.data.Dataset):
    def __init__(self, source_dset, sampled_idx):
        self.source_dset = source_dset
        self.sampled_idx = sampled_idx

    def __len__(self):
        return len(self.sampled_idx)
    
    def __getitem__(self, idx):
        return self.source_dset[self.sampled_idx[idx]]
    

def update_meters(meters, metrics):
    for key in meters.keys():
        for metric_name in meters[key].keys():
            meters[key][metric_name].update(metrics[key][metric_name])

def extract_meters(meters):
    return {
        key : {
            metric_name : meter.avg for metric_name, meter in metrics.items()
        } for key, metrics in meters.items()
    }


class Evaluator:

    def __init__(self, train_scenes, test_scenes, train_injection_dataset, test_injection_dataset, pose_sampler, config, repo_path) : 
        self.train_scenes = train_scenes
        self.test_scenes = test_scenes
        self.train_injection_dataset = train_injection_dataset
        self.test_injection_dataset = test_injection_dataset
        self.pose_sampler = pose_sampler

        self.psnr_func = PeakSignalNoiseRatio(data_range=(-1.0, 1.0))
        self.ssim_func = partial(structural_similarity_index_measure, data_range=(-1.0, 1.0))
        self.lpips_func = LearnedPerceptualImagePatchSimilarity(normalize=False) # input is in [-1, 1] so normalize should be False
        self.mse_func = torch.nn.functional.mse_loss

        self.config = config
        if 'exp_name' in self.config.keys():
            self.exp_name = self.config.exp_name
        elif 'exploit_exp_name' in self.config.keys():
            self.exp_name = self.config.exploit_exp_name
        else:
            raise ValueError("No experiment name found in config")
        self.repo_path = repo_path

        if train_injection_dataset:
            sampled_idxs = random.sample(range(len(train_injection_dataset)), min(len(train_injection_dataset), self.config.eval.injection_dashboard.n_img))
            self.train_injection_dataset_sampled = IdxSampledDataset(train_injection_dataset, sampled_idxs)

    @torch.no_grad()
    def compute_metrics(self, im1:torch.Tensor, im2:torch.Tensor, include_psnr=True, include_mse=False, include_lpips=False, include_ssim=False):
        "Compute the PSNR between two tensor images, expected with shape [... C, H, W] and values in [-1, 1] "

        metrics = {}

        if include_mse:
            metrics['mse'] = self.mse_func(im1, im2).item()
        if include_psnr:
            metrics['psnr'] = self.psnr_func(im1, im2).item()
        if include_lpips:
            metrics['lpips'] = self.lpips_func(im1, im2).item()
        if include_ssim:
            metrics['ssim'] = self.ssim_func(im1, im2).item()
        return metrics
    
    @torch.no_grad()
    def _get_lnerf_metrics(self, vae, latent_nerf, renderer, single_scene_data, normalizer, batch_size, device):
        
        self.lpips_func.to(device)
        self.psnr_func.to(device)
        latent_nerf = latent_nerf.to(device)
        
        meters = {
            'metrics_e' : {
                'psnr': AverageMeter(),
            },
            'metrics_d' : {
                'psnr': AverageMeter(),
                'lpips': AverageMeter(),
                'ssim': AverageMeter(),
            },
            'metrics_ae' : {
                'psnr': AverageMeter(),
                'lpips': AverageMeter(),
                'ssim': AverageMeter(),
            },
            'metrics_d-ae' : {
                'psnr': AverageMeter(),
                'lpips': AverageMeter(),
                'ssim': AverageMeter(),
            }
        }
        
        loader = torch.utils.data.DataLoader(single_scene_data, batch_size=batch_size, shuffle=False)
        for i, batch in enumerate(loader) :
            # getting g.t. image and pose
            rgb_img = batch['img'].to(device)
            pose = batch['pose'].to(device).unsqueeze(0)

            # getting pseudo g.t. latent img
            latent_img = vae.encode(rgb_img).latent_dist.sample()
            normalized_latent_img = normalizer.normalize(latent_img)

            # getting reconstructed latent img 
            rendering = renderer(latent_nerf.unsqueeze(0), pose)
            normalized_latent_img_render = rendering['img'].squeeze(0).clamp(-1,1)
            latent_img_render = normalizer.denormalize(normalized_latent_img_render)

            # getting reconstruced rgb img
            rgb_img_render = vae.decode(latent_img_render).sample.clamp(-1,1)

            # getting autoencoded rgb img
            rgb_img_ae = vae.decode(latent_img).sample.clamp(-1,1)
            
            metrcis = {
                'metrics_e' : self.compute_metrics(
                    normalized_latent_img, normalized_latent_img_render,
                    include_psnr=True, include_lpips=False, include_ssim=False
                ),
                'metrics_d' : self.compute_metrics(
                    rgb_img, rgb_img_render,
                    include_psnr=True, include_lpips=True, include_ssim=True
                ),
                'metrics_ae' : self.compute_metrics(
                    rgb_img, rgb_img_ae,
                    include_psnr=True, include_lpips=True, include_ssim=True
                ),
                'metrics_d-ae' : self.compute_metrics(
                    rgb_img_ae, rgb_img_render,
                    include_psnr=True, include_lpips=True, include_ssim=True
                )
            }

            update_meters(meters, metrcis)

        return extract_meters(meters)

    @torch.no_grad()
    def _get_rgb_override_metrics(self, vae, nerf, renderer, single_scene_data, normalizer, batch_size, device):

        self.lpips_func.to(device)
        self.psnr_func.to(device)
        nerf = nerf.to(device)

        meters = {
            'metrics_rgb_override' : {
                'psnr': AverageMeter(),
                'lpips': AverageMeter(),
                'ssim': AverageMeter(),
            },
        }

        loader = torch.utils.data.DataLoader(single_scene_data, batch_size=batch_size, shuffle=False)

        for i, batch in enumerate(loader) :
            # getting g.t. image and pose
            img = batch['img'].to(device)
            pose = batch['pose'].to(device).unsqueeze(0)

            # getting reconstructed latent img 
            rendering = renderer(nerf.unsqueeze(0), pose)
            img_render = rendering['img'].squeeze(0).clamp(-1,1)
            
            metrics = {
                'metrics_rgb_override' : self.compute_metrics(
                    img, img_render,
                    include_psnr=True, include_lpips=True, include_ssim=True
                )
            }

            update_meters(meters, metrics)

        return extract_meters(meters)
    
    @torch.no_grad()
    def get_consistency_metrics(self, vae, latent_nerfs, renderer, normalizer, batch_size, device):

        metrics = {}
        if self.config.rgb_override:
            get_metrics_fn = self._get_rgb_override_metrics
        else: 
            get_metrics_fn = self._get_lnerf_metrics

        for scene_idx in tqdm.tqdm(range(len(latent_nerfs)), desc="Computing evaluation metrics") :
            metrics[f'scene_{scene_idx}'] = {
                'train' : get_metrics_fn(vae, latent_nerfs[scene_idx], renderer, self.train_scenes[scene_idx], normalizer, batch_size, device),
                'test' : get_metrics_fn(vae, latent_nerfs[scene_idx], renderer, self.test_scenes[scene_idx], normalizer, batch_size, device)
            }
        
        mean_metric_dict = {}
        for split in list(metrics.values())[0].keys() :
            mean_metric_dict[split] = {}
            for model in metrics['scene_0'][split].keys() :
                mean_metric_dict[split][model] = {}
                for metric_name in metrics['scene_0'][split][model].keys() :
                    mean_metric_dict[split][model][metric_name] = np.mean([
                        metrics[f'scene_{scene_idx}'][split][model][metric_name] for scene_idx in range(len(latent_nerfs))
                    ])

        if not self.config.eval.metrics.log_scenes_independently : 
            return mean_metric_dict
        
        metrics['all_scenes'] = mean_metric_dict
        
        return metrics
    
    @torch.no_grad()
    def get_video_eval(self, vae, latent_nerfs, renderer, normalizer, batch_size, device):
        """Makes a video of the given latent nerfs
        Returns the wandb.Video object"""

        vae.to(device)
        
        latent_to_pil_transform = tr.Compose([
            tr.Normalize(-1, 2),
            tr.Lambda(lambda x : x.clamp(0, 1)),
            #tr.Lambda(lambda x : x[:3]),
            tr.ToPILImage()
        ])

        if self.config.rgb_override:
            latent_to_rgb_transform = latent_to_pil_transform

        else: 
            def _decode_latent(latent_img):
                latent_img.to(device)
                latent_img = normalizer.denormalize(latent_img.unsqueeze(0))
                rgb_img = vae.decode(latent_img).sample.clamp(-1,1).squeeze(0)
                return rgb_img
            
            
            latent_to_rgb_transform = tr.Compose([
                tr.Lambda(_decode_latent),
                tr.Lambda(lambda x: (x + 1)/2),
                tr.ToPILImage()
            ])

            
        rd_string = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        video_savename = f"latent_nerf_video_{rd_string}.mp4"
        save_dir = os.path.join(self.repo_path, self.config.savedir, self.exp_name, 'buffer')
        os.makedirs(save_dir, exist_ok=True)
        make_dashboard_video(
            renderer, 
            latent_nerfs, 
            self.pose_sampler, 
            latent_to_pil_transform, 
            latent_to_rgb_transform, 
            self.config.latent_nerf.rendering_options, 
            batch_size, 
            device,
            video_savename, 
            save_dir=save_dir, 
            n_frames=self.config.eval.video.n_frames,
            fps=self.config.eval.video.fps,
            azimuth_range=self.config.eval.video.azimuth_range,
            elevation_range=self.config.eval.video.elevation_range,
            radius_range=self.config.eval.video.radius_range
        )
        
        return wandb.Video(os.path.join(self.repo_path, self.config.savedir, self.exp_name, 'buffer', video_savename))
    
    @torch.no_grad()
    def get_scene_views(self, vae, latent_nerf, renderer, single_scene_data, normalizer, batch_size, device, n_views):
        latent_nerf = latent_nerf.to(device)

        all_rgb_img = []
        all_latent_img = []
        all_latent_img_render = []
        all_rgb_img_render = []
        all_rgb_img_ae = []

        batch_size = min(n_views, batch_size)
        n_loop = math.ceil(n_views / batch_size)

        data_iterator = iter(torch.utils.data.DataLoader(single_scene_data, batch_size=batch_size, shuffle=False))

        for i in range(n_loop):
            batch = next(data_iterator)

            # getting g.t. image and pose
            rgb_img = batch['img'].to(device)
            pose = batch['pose'].to(device).unsqueeze(0)

            # getting pseudo g.t. latent img
            latent_img = vae.encode(rgb_img).latent_dist.sample()
            normalized_latent_img = normalizer.normalize(latent_img)

            # getting reconstructed latent img 
            rendering = renderer(latent_nerf.unsqueeze(0), pose)
            normalized_latent_img_render = rendering['img'].squeeze(0).clamp(-1,1)
            latent_img_render = normalizer.denormalize(normalized_latent_img_render)

            # getting reconstruced rgb img
            rgb_img_render = vae.decode(latent_img_render).sample.clamp(-1,1)

            # getting autoencoded rgb img
            rgb_img_ae = vae.decode(latent_img).sample.clamp(-1,1)

            # append
            all_rgb_img.append(rgb_img)
            all_latent_img.append(normalized_latent_img)
            all_latent_img_render.append(normalized_latent_img_render)
            all_rgb_img_render.append(rgb_img_render)
            all_rgb_img_ae.append(rgb_img_ae)

        all_rgb_img = torch.concatenate(all_rgb_img, dim=0)[:n_views]
        all_latent_img = torch.concatenate(all_latent_img, dim=0)[:n_views]
        all_latent_img_render = torch.concatenate(all_latent_img_render, dim=0)[:n_views]
        all_rgb_img_render = torch.concatenate(all_rgb_img_render, dim=0)[:n_views]
        all_rgb_img_ae = torch.concatenate(all_rgb_img_ae, dim=0)[:n_views]

        return {
            'rgb_img' : all_rgb_img,
            'latent_img' : all_latent_img,
            'latent_img_render' : all_latent_img_render,
            'rgb_img_render' : all_rgb_img_render,
            'rgb_img_ae' : all_rgb_img_ae
        }

    @torch.no_grad()
    def get_consistency_dashboard(self, epoch, vae, latent_nerfs, renderer, normalizer, batch_size, device):
        """
        This function makes an "dashboard" plot where we can see, for each scene, for some views:
        - the rgb ground truth
        - the latent nerf renderings
        - the latent pseudo ground truth 
        - the rgb decoded render
        - the autoencoded rgb image

        It also evaluates the autoencoded image on the test set of the injected dataset
        """

        all_figures = {
            'consistency_dashboard/train/' : {},
            'consistency_dashboard/test/' : {}
        }

        for scene_idx, latent_nerf in enumerate(latent_nerfs[:self.config.eval.dashboard.n_scenes]):
            scene_train_views = self.get_scene_views(vae, latent_nerf, renderer, self.train_scenes[scene_idx], normalizer, batch_size, device, n_views=self.config.eval.dashboard.n_views)
            scene_test_views = self.get_scene_views(vae, latent_nerf, renderer, self.test_scenes[scene_idx], normalizer, batch_size, device, n_views=self.config.eval.dashboard.n_views)
            all_figures['consistency_dashboard/train/'][f'scene_{scene_idx}'] = make_dashboard(epoch, scene_train_views, self.config.eval.dashboard.n_views)
            all_figures['consistency_dashboard/test/'][f'scene_{scene_idx}'] = make_dashboard(epoch, scene_test_views, self.config.eval.dashboard.n_views)

        plt.close("all")
        return all_figures
    
    @torch.no_grad()
    def get_injection_images(self, vae, dset, normalizer, batch_size, device, n_img):

        batch_size = min(n_img, batch_size)
        n_loop = math.ceil(n_img / batch_size)
        data_iterator = iter(torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False))

        gt_imgs = []
        recon_imgs = []
        latent_imgs = []
        for i in range(n_loop):
            batch = next(data_iterator)
            img = batch['img'].to(device)
            
            # forward
            latent_img = vae.encode(img).latent_dist.sample()
            recon_img = vae.decode(latent_img).sample.clamp(-1,1)

            gt_imgs.append(img)
            latent_imgs.append(normalizer.normalize(latent_img))
            recon_imgs.append(recon_img)

        return {
            'gt_imgs' : torch.cat(gt_imgs, dim=0)[:n_img],
            'latent_imgs' : torch.cat(latent_imgs, dim=0)[:n_img],
            'recon_imgs' : torch.cat(recon_imgs, dim=0)[:n_img],
        }

    @torch.no_grad()
    def get_injection_dashboard(self, epoch, vae, normalizer, batch_size, device):

        img_train = self.get_injection_images(vae, self.train_injection_dataset, normalizer, batch_size, device, self.config.eval.injection_dashboard.n_img)
        img_test = self.get_injection_images(vae, self.test_injection_dataset, normalizer, batch_size, device, self.config.eval.injection_dashboard.n_img)
        
        fig_train = make_injection_dashboard(epoch, img_train, self.config.eval.injection_dashboard.n_img)
        fig_test = make_injection_dashboard(epoch, img_test, self.config.eval.injection_dashboard.n_img)

        plt.close("all")
        return {
            'injection_dashboard/train' : fig_train,
            'injection_dashboard/test' : fig_test
        }
        
    @torch.no_grad()
    def _get_injection_metrics(self, vae, dset, batch_size, device):

        self.lpips_func.to(device)
        self.psnr_func.to(device)

        meters = {
            'metrics_ae': {
                'psnr': AverageMeter(),
                'lpips': AverageMeter(),
                'ssim': AverageMeter(),
            }
        }

        loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)
        for i, batch in enumerate(loader):
            img = batch['img'].to(device)
            
            # forward
            latent_img = vae.encode(img).latent_dist.sample()
            recon_img = vae.decode(latent_img).sample.clamp(-1,1)

            metrics = {
                "metrics_ae" : self.compute_metrics(
                    img, recon_img,
                    include_psnr=True, include_lpips=True, include_ssim=True
                )
            }

            update_meters(meters, metrics)

        return extract_meters(meters)
    
    @torch.no_grad()
    def get_injection_metrics(self, vae, batch_size, device):
        return {
            'train' : self._get_injection_metrics(vae, self.train_injection_dataset_sampled, batch_size, device),
            'test' : self._get_injection_metrics(vae, self.test_injection_dataset, batch_size, device)
        }