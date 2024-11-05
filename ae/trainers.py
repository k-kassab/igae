import os, sys
import torch
import wandb
import time
import pickle
import itertools
import math
import tqdm
import diffusers
import copy
from prodict import Prodict
from typing import Union
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from ae.utils import AverageMeter, compute_tv, save_config
from ae.triplane_renderer import TriPlaneRenderer
from ae.camera_utils import LazyPoseSampler
from ae.utils import do_now, set_requires_grad, to_dict
from datasets.dataset import DataReader, ObjectDataset, MultiSceneDataset, CachedDataset, ImageDataset, MyIterableDataset
from ae.global_triplanes import TriplaneManager


class BaselineVaeManager:

    # note: 
    # bg_color is in [0, 1] range
    # even though nerfs are rendered in [-1, 1] range
    VAE_CONFIGS = {
        'runwayml/stable-diffusion-v1-5': {
            'n_latent_channels' : 4,
            'downscale_factor' : 8,
            'bg_color': [0.6176, 0.5711, 0.5046, 0.4422],
            'instantiation_kwargs' : {
                'subfolder' : 'vae',
                'revision' : 'main',
            }
        },
        'ostris/vae-kl-f8-d16' : {
            'n_latent_channels' : 16,
            'downscale_factor' : 8,
            'bg_color': [0.4090, 0.5550, 0.4015, 0.5846, 0.4492, 0.5380, 0.5314, 0.4108, 0.5015,
        0.6012, 0.4245, 0.5039, 0.4920, 0.4955, 0.5296, 0.4999],
            'instantiation_kwargs' : {
                #'torch_dtype' : torch.float16,
            },
        },
    }

    def __init__(self, vae_name):
        self.vae_name = vae_name

    def get_vae(self):
        kwargs = self.VAE_CONFIGS[self.vae_name]['instantiation_kwargs']
        return diffusers.AutoencoderKL.from_pretrained(
            self.vae_name, **kwargs
        )
    
    def get_bg_color(self):
        return self.VAE_CONFIGS[self.vae_name]['bg_color']

    def get_num_latent_channels(self):
        return self.VAE_CONFIGS[self.vae_name]['n_latent_channels']
    
    def get_downscale_factor(self):
        return self.VAE_CONFIGS[self.vae_name]['downscale_factor']

def unwrap_models(*models, accelerator) :
    return [accelerator.unwrap_model(model) for model in models]

def save_models(savename, savedir, vae, nerfs, renderer, include_global_nerfs=False): 
    dict_to_save = {
        'vae': vae.state_dict(),
        'nerfs': nerfs.state_dict(),
        'renderer': renderer.state_dict(),
    }
    if include_global_nerfs:
        dict_to_save['global_nerfs'] = nerfs.global_planes.data

    torch.save(dict_to_save, os.path.join(savedir, savename))

def use_encoder_in_consistency(loss_ae, loss_e, loss_d, loss_rgb_override):
    """args:
        loss_ae: bool
        loss_e: bool
        loss_d: bool 
    """
    return loss_e or loss_ae

def use_decoder_in_consistency(loss_ae, loss_e, loss_d, loss_rgb_override):
    """args:
        loss_ae: bool
        loss_e: bool
        loss_d: bool 
    """
    return loss_d or loss_ae

def use_nerfs_in_consistency(loss_ae, loss_e, loss_d, loss_rgb_override):
    """args:
        loss_e: bool
    """
    return loss_e or loss_d or loss_rgb_override

def multiassign(d, keys, values):
    d.update(zip(keys, values))

class NerfVAE(torch.nn.Module):

    def __init__(
            self,
            vae,
            latent_renderer,
            latent_nerfs,
            normalizer
            ):
        super().__init__()

        # nn submodules
        self.vae = vae
        self.latent_renderer = latent_renderer
        self.latent_nerfs =latent_nerfs

        self.normalizer = normalizer
        self.mse_loss = torch.nn.functional.mse_loss
        self.perceptual_loss = LearnedPerceptualImagePatchSimilarity(normalize=False)
    
    def get_consistency_loss_dict(self, rgb_img, pose, scene_idxs, additional_args, cached_latents=None):
        """Returns the computed loss for the vae and the nerfs
        Args:
            rgb_img: torch.FloatTensor[B, 3, H, W]:
            pose: torch.FloatTensor[B, 25]
            scene_idxs: torch.LongTensor[B]
            cached_latents: torch.FloatTensor[B, 4, H, W] or None
            additional_args: prodict

        Example:
            additional_args = {
                "mu_nerf" : False
                "losses": {
                    loss_e: True,
                    loss_d: True,
                    loss_ae: True,
                    loss_rgb_override: True
                }
                "regularizations": {
                    tv: True,
                    tv_mode: "l2",
                    depth_tv: False,
                    depth_tv_mode: "l2"
                }
            }

        Returns: 
            computed_losses: dict

        Example:
            computed_losses = {
                "loss" : torch.Tensor
                "loss_ae" : torch.Tensor
                "loss_e" : torch.Tensor
                "loss_d" : torch.Tensor
                "tv" : torch.Tensor
                "depth_tv" : torch.Tensor
            }
        """

        use_encoder = use_encoder_in_consistency(**additional_args.losses)
        use_decoder = use_decoder_in_consistency(**additional_args.losses)

        if use_encoder:
            if cached_latents:
                latent_img = cached_latents
            else : 
                latent_dist = self.vae.encode(rgb_img).latent_dist
                if additional_args.mu_nerf:
                    latent_img = latent_dist.mean
                else : 
                    latent_img = latent_dist.sample()

            #3. Normalize latent img z
            normalized_latent_img = self.normalizer.normalize(latent_img) 

            if use_decoder:
                rgb_img_ae = self.vae.decode(latent_img).sample.clamp(-1,1)

        #4. Latent nerf rendering
        pose = pose.unsqueeze(1) #shape [bs, n_renders(=1), 25]
        latent_nerfs = self.latent_nerfs[scene_idxs].squeeze(1)
        rendering = self.latent_renderer(latent_nerfs, pose)
        normalized_latent_img_render = rendering['img'].squeeze(1).clamp(-1, 1)
        latent_depth = rendering['img_depth'].squeeze(1)

        #5. Decode latent rendering into rgb domain
        if use_decoder:
            latent_img_render = self.normalizer.denormalize(normalized_latent_img_render)
            rgb_img_render = self.vae.decode(latent_img_render).sample.clamp(-1,1)

        #6. Compute losses and optimize
        computed_losses = {} 
        if additional_args.losses.loss_ae:
            loss_ae = self.mse_loss(rgb_img_ae, rgb_img)
            computed_losses['loss_ae'] = loss_ae

        if additional_args.losses.loss_e:
            loss_e = self.mse_loss(normalized_latent_img, normalized_latent_img_render)
            computed_losses['loss_e'] = loss_e

        if additional_args.losses.loss_d:
            loss_d = self.mse_loss(rgb_img_render, rgb_img)
            computed_losses['loss_d'] = loss_d

        if additional_args.losses.loss_rgb_override:
            loss_rgb_override = self.mse_loss(normalized_latent_img_render, rgb_img)
            computed_losses['loss_rgb_override'] = loss_rgb_override
            
        #6.1 Add eventual regularisation terms
        
        # TV regularization for Tri-Planes features
        if additional_args.regularizations.tv: 
            tv = compute_tv(latent_nerfs, *additional_args.regularizations.tv_mode) 
            computed_losses['tv'] = tv

        # TV regularization for depths    
        if additional_args.regularizations.depth_tv: 
            depth_tv = compute_tv(latent_depth, *additional_args.regularizations.depth_tv_mode) 
            computed_losses['depth_tv'] = depth_tv

        #loss_meter.update(loss.item())

        return computed_losses

    def get_injection_loss_dict(self, rgb_img, additional_args):
        """Returns the vae loss provided the input image
        Args:  
            rgb_img: torch.Tensor[B, C, H, W]
            additional_args: prodict
        Example:
            additionale_args = {
                "kl_div" : False
            }

            additional_args = {
                "mu_nerf" : False
                "losses": {
                    loss_mse: True,
                    loss_lpip: True,
                }
                "regularizations": {
                    kl_div: False,
                    "tv": False,
                }
            }


        Returns: 
            computed_losses: dict

        Example:
            computed_losses = {
                "loss_ae" : torch.Tensor
                "loss_ae_perceptual" : torch.Tensor
                "kl_div" : torch.Tensor
            }
        """

        # 1. Forward thourgh vae
        latent_dist = self.vae.encode(rgb_img).latent_dist
        latent_img = latent_dist.sample()
        rgb_img_ae = self.vae.decode(latent_img).sample.clamp(-1,1)

        #2. Compute loss
        computed_losses = {}
        if additional_args.losses.loss_mse:
            computed_losses['loss_mse'] = self.mse_loss(rgb_img_ae, rgb_img)

        if additional_args.losses.loss_lpip:
            computed_losses['loss_lpip'] = self.perceptual_loss(rgb_img_ae, rgb_img)

        #3. Compute regularizations
        if additional_args.regularizations.kl_div:
            raise NotImplementedError("KL divergence regularization not implemented yet")
        
        if additional_args.regularizations.tv:
            computed_losses['tv'] = compute_tv(latent_img, *additional_args.regularizations.tv_mode)

        return computed_losses

    def forward(self, batch_consistency=None, batch_injection=None, consistency_args=None,  injection_args=None) : 

        computed_losses = {}

        if batch_consistency:
            assert consistency_args, "Consistency args should be provided when batch_consistency is not None"

            computed_losses['consistency'] =  self.get_consistency_loss_dict(
                    rgb_img=batch_consistency['img'],
                    pose=batch_consistency['pose'],
                    scene_idxs=batch_consistency['scene_idx'],
                    cached_latents=batch_consistency.get('cached_latents', None),
                    additional_args=consistency_args
                )

        if batch_injection:
            assert injection_args, "Injection args should be provided when batch_injection is not None"
            computed_losses['injection'] = self.get_injection_loss_dict(
                rgb_img=batch_injection['img'],
                additional_args=injection_args
            )

        return computed_losses
            

class Trainer:
    def __init__(
            self, 
            config, 
            t_args,
            expname, 
            accelerator,
            multi_scene_set,
            injection_set,
            nerf_vae, 
            normalizer,
            evaluator,
            repo_path,
            debug=False
            ):
        
        # args
        self.config = config
        self.t_args = t_args
        self.expname = expname
        self.repo_path = repo_path
        self.debug = debug

        # state
        self.use_encoder = (
            use_encoder_in_consistency(**t_args.consistency.losses)
            or 
            self.t_args.injection.apply
        )
        self.train_encoder = (
            self.use_encoder
            and
            not self.t_args.freezes.freeze_encoder
        )
        self.use_decoder = (
            use_decoder_in_consistency(**t_args.consistency.losses) 
            or 
            self.t_args.injection.apply
        )
        self.train_decoder = (
            self.use_decoder
            and
            not self.t_args.freezes.freeze_decoder
        ) 
        self.use_nerfs =  (
            use_nerfs_in_consistency(**t_args.consistency.losses)   
        )
        self.train_nerfs = (
            self.use_nerfs
            and
            not (self.t_args.freezes.freeze_lnerf and self.t_args.freezes.freeze_base_coefs and self.t_args.freezes.freeze_global_lnerf)
        ) 

        # model
        self.nerf_vae = nerf_vae
        self.normalizer = normalizer
        self.evaluator = evaluator
        self.accelerator = accelerator

        # dataloaders
        self.dataloader_consistency = self.get_consistency_dataloader(multi_scene_set)
        self.dataloader_injection = self.get_injection_dataloader(injection_set)

        # optimizers and schedulers
        self.optimizers, self.schedulers = self.get_optimizers_schedulers()

        # Freezing models if necessary
        if self.t_args.freezes.freeze_encoder:
            set_requires_grad(self.nerf_vae.vae.encoder, False)
        if self.t_args.freezes.freeze_decoder:
            set_requires_grad(self.nerf_vae.vae.decoder, False)
        if self.t_args.freezes.freeze_lnerf:
            set_requires_grad(self.nerf_vae.latent_nerfs.local_planes, False)
            set_requires_grad(self.nerf_vae.latent_renderer, False)
        if self.config.global_latent_nerf.apply :
            if self.t_args.freezes.freeze_global_lnerf:
                set_requires_grad(self.nerf_vae.latent_nerfs.global_planes, False)
            if self.t_args.freezes.freeze_base_coefs:
                set_requires_grad(self.nerf_vae.latent_nerfs.coefs, False)

        # Prepare all modules
        self.prepare_modules()
        self.iterator_injection = itertools.cycle(self.dataloader_injection)

        # logging
        self.meters = {
            "total_loss" : AverageMeter(),
            "consistency" : {
                "consistency_loss" : AverageMeter(),
                "loss_ae" : AverageMeter(),
                "loss_e" : AverageMeter(),
                "loss_d" : AverageMeter(),
                "loss_rgb_override" : AverageMeter(),
                "tv" : AverageMeter(),
                "depth_tv" : AverageMeter(),
            },
            "injection" : {
                "injection_loss" : AverageMeter(),
                "loss_mse": AverageMeter(),
                "loss_lpip": AverageMeter(),
                "kl_div": AverageMeter(),
                "tv" : AverageMeter(),
            },
        } 
        self.cum_train_time = 0
        self.cum_eval_time = 0

        if self.accelerator.is_main_process and self.config.wandb.apply:
            wandb.init(
                entity=self.config.wandb.entity,
                project=self.config.wandb.project_name,
                config=copy.deepcopy(self.config),
                dir=self.repo_path+"/wandb",
                notes=self.t_args.wandb_note,
                name=self.expname
            )

    def get_consistency_dataloader(self, multi_scene_set):
        if self.use_encoder and self.t_args.cache_latents.apply: 
            assert self.t_args.freezes.freeze_encoder, "Caching latents should only be used with a frozen encoder"
            if self.accelerator.is_local_main_process:
                multi_scene_set = CachedDataset(
                    multi_scene_set, 
                    self.accelerator.unwrap_model(self.nerf_vae).vae, 
                    self.accelerator.device, 
                    self.t_args.cache_latents.batch_size, 
                    self.t_args.cache_latents.use_mean, 
                    self.repo_path, 
                    self.config.savedir, 
                    self.expname
                )
            self.accelerator.wait_for_everyone()

        consistency_dataloader = torch.utils.data.DataLoader(multi_scene_set, batch_size=self.t_args.consistency.batch_size, shuffle=True)
        return consistency_dataloader

    def get_injection_dataloader(self, injection_set):
        assert len(injection_set) > 0
        injection_dataloader = torch.utils.data.DataLoader(injection_set, batch_size=self.t_args.injection.batch_size) # Iterator style Dataloader
        return injection_dataloader

    def get_optimizers_schedulers(self):
        "To be called before the model is prepared"

        # Scheduler config
        if self.t_args.optim.scheduler.type == 'multistep': 
            Scheduler = torch.optim.lr_scheduler.MultiStepLR
            scheduler_kwargs = self.t_args.optim.scheduler.multistep_config
        elif self.t_args.optim.scheduler.type  == 'exp':
            Scheduler = torch.optim.lr_scheduler.ExponentialLR
            scheduler_kwargs = self.t_args.optim.scheduler.exp_config
        else: 
            raise ValueError(f"Scheduler type {self.t_args.optim.scheduler.type} not recognized")
        

        # Optimizer config:
        lr_scale = 1.0
        if self.t_args.optim.scale_lr:
            lr_scale = self.accelerator.num_processes

        # Instanciating optimizers and schedulers
        optimizers = {}
        schedulers = {}


        # 1. Latent nerfs
        if self.use_nerfs:
            #1.a Local latent nerfs
            optimizer_latent_nerfs = torch.optim.Adam([
                {'params' : self.nerf_vae.latent_renderer.parameters(), 'lr' : lr_scale * self.t_args.optim.latent_nerf.tinymlp_lr},
                {'params' : self.nerf_vae.latent_nerfs.local_planes,    'lr' : lr_scale * self.t_args.optim.latent_nerf.lr},
            ])

            scheduler_latent_nerfs = Scheduler(
                optimizer_latent_nerfs, 
                **scheduler_kwargs
            )

            optimizers['latent_nerfs'] = optimizer_latent_nerfs
            schedulers['latent_nerfs'] = scheduler_latent_nerfs

            # 1.b Global latent nerfs
            if self.config.global_latent_nerf.apply :
                optimizer_global_latent_nerfs = torch.optim.Adam([{
                    'params' : self.nerf_vae.latent_nerfs.global_planes,                 
                    'lr' : lr_scale * self.t_args.optim.global_latent_nerf.lr
                }])
                scheduler_global_latent_nerfs = Scheduler(
                    optimizer_global_latent_nerfs, 
                    **scheduler_kwargs
                )
                optimizer_base_coefs = torch.optim.Adam([{
                    'params' : self.nerf_vae.latent_nerfs.coefs,                 
                    'lr' : lr_scale * self.t_args.optim.base_coefs.lr
                }])
                scheduler_base_coefs = Scheduler(
                    optimizer_base_coefs, 
                    **scheduler_kwargs
                )

                optimizers['global_latent_nerfs'] = optimizer_global_latent_nerfs
                schedulers['global_latent_nerfs'] = scheduler_global_latent_nerfs
                optimizers['base_coefs'] = optimizer_base_coefs
                schedulers['base_coefs'] = scheduler_base_coefs

        # 2. Encoder
        if self.use_encoder:
            encoder_param = filter(lambda p: p.requires_grad, self.nerf_vae.vae.encoder.parameters())
            optimizer_encoder = torch.optim.Adam([{
                'params' : encoder_param, 
                'lr' : lr_scale * self.t_args.optim.encoder.lr
            }])
            scheduler_encoder = Scheduler(
                optimizer_encoder, 
                **scheduler_kwargs
            ) 
            optimizers['encoder'] = optimizer_encoder
            schedulers['encoder'] = scheduler_encoder

        # 3. Decoder
        if self.use_decoder:
            decoder_param = filter(lambda p: p.requires_grad, self.nerf_vae.vae.decoder.parameters())
            optimizer_decoder = torch.optim.Adam([{
                'params' : decoder_param, 
                'lr' : lr_scale * self.t_args.optim.decoder.lr
            }])
            scheduler_decoder = Scheduler(
                optimizer_decoder, 
                **scheduler_kwargs
            )
            optimizers['decoder'] = optimizer_decoder
            schedulers['decoder'] = scheduler_decoder

        return optimizers, schedulers

    def prepare_modules(self):
        self.nerf_vae, self.dataloader_consistency, self.dataloader_injection, *new_optimizers = self.accelerator.prepare(self.nerf_vae, self.dataloader_consistency, self.dataloader_injection, *self.optimizers.values())
        multiassign(self.optimizers, self.optimizers.keys(), new_optimizers)

    def get_log_dict_from_meters(self):
        log_dict = {}
        for meter_name, meter in self.meters.items():
            if isinstance(meter, dict):
                for sub_meter_name, sub_meter in meter.items():
                    log_dict[f"{meter_name}/{sub_meter_name}"] = sub_meter.avg
            else:
                log_dict[meter_name] = meter.avg
        return log_dict

    def reset_meters(self):
        for meter_or_dict in self.meters.values():
            if isinstance(meter_or_dict, dict):
                for sub_meter in meter_or_dict.values():
                    sub_meter.reset()
            else:
                meter_or_dict.reset()

    def get_total_loss(self, batch_consistency=None, batch_injection=None):
        """Returns a total loss providing input data for NerfVAE.

        This function:
            1. computes individual losses
            2. logs them to the meters
            3. Sums them accoring to the weights specified in t_args.optim.loss_coefs
            4. Returns the total loss

        Input arguments can either be:
            - only consistency data
            - only injection data
            - both consistency and injection data
        The total loss will be computed accordingly.
        """

        consistency_args = Prodict.from_dict({
            "losses": self.t_args.consistency.losses,
            "regularizations": self.t_args.consistency.regularizations,
            "mu_nerf": self.config.latent_nerf.mu_nerf,
        })
        injection_args = Prodict.from_dict({
            "losses":self.t_args.injection.losses,
            "regularizations": self.t_args.injection.regularizations,
        })

        losses = self.nerf_vae(batch_consistency, batch_injection, consistency_args, injection_args)

        total_loss = 0
        for loss_type in losses.keys():

            current_loss = 0
            for loss_name, loss_value in losses[loss_type].items():
                current_loss += self.t_args[loss_type]['weights'][loss_name] * loss_value
                self.meters[loss_type][loss_name].update(loss_value.item())

            self.meters[loss_type][f"{loss_type}_loss"].update(current_loss.item())
            total_loss += current_loss

        self.meters['total_loss'].update(total_loss.item()) 
        return total_loss

    def step_relevant_optimizers(self, consistency=True, injection=True):
        
        # Note: 
        # injection_iter => (use_encoder and use_decoder) 

        consistency_only = consistency and not injection

        if self.train_encoder:
            self.optimizers['encoder'].step()
            self.optimizers['encoder'].zero_grad()

        if self.train_decoder:
            self.optimizers['decoder'].step()
            self.optimizers['decoder'].zero_grad()

        if consistency_only:
            if self.train_nerfs:
                if not self.t_args.freezes.freeze_lnerf:
                    self.optimizers['latent_nerfs'].step()
                    self.optimizers['latent_nerfs'].zero_grad()

                if self.config.global_latent_nerf.apply:
                    if not self.t_args.freezes.freeze_global_lnerf :
                        self.optimizers['global_latent_nerfs'].step()
                        self.optimizers['global_latent_nerfs'].zero_grad()

                    if not self.t_args.freezes.freeze_base_coefs :
                        self.optimizers['base_coefs'].step()
                        self.optimizers['base_coefs'].zero_grad()

    def step_relevant_schedulers(self):
        """Step all relevant schedulers according to the current training scheme. 
        To be used at the end of each training EPOCH"""

        if self.train_encoder:
            self.schedulers['encoder'].step()

        if self.train_decoder:
            self.schedulers['decoder'].step()

        if self.train_nerfs:
            if not self.t_args.freezes.freeze_lnerf:
                self.schedulers['latent_nerfs'].step()

            if self.config.global_latent_nerf.apply:
                if not self.t_args.freezes.freeze_global_lnerf :
                    self.schedulers['global_latent_nerfs'].step() 

                if not self.t_args.freezes.freeze_base_coefs :
                    self.schedulers['base_coefs'].step()

    def forward_backward_step(self, batch_consistency=None, batch_injection=None):
        loss = self.get_total_loss(batch_consistency, batch_injection)
        self.accelerator.backward(loss)

        self.step_relevant_optimizers(
            consistency=batch_consistency is not None, 
            injection=batch_injection is not None
        ) 

    def consistency_iter(self, batch_consistency):
        self.forward_backward_step(
            batch_consistency=batch_consistency,
            batch_injection=None
        )

    def injection_iter(self, batch_injection):
        self.forward_backward_step(
            batch_consistency=None,
            batch_injection=batch_injection
        )
    
    def consistency_and_injection_iter(self, batch_consistency, batch_injection, strategy):
        if strategy == 'joint':
            self.forward_backward_step(
                batch_consistency=batch_consistency,
                batch_injection=batch_injection
            )
        elif strategy == 'alternating':
            self.consistency_iter(batch_consistency)
            self.injection_iter(batch_injection)
        else:
            raise ValueError(f"Unknown injection strategy {self.t_args.injection.strategy}. Should be 'alternating' or 'joint'")     
    
    def train(self) :
        
        # I. Intial evaluation
        if self.t_args.logging.initial_eval and self.config.wandb.apply:
            if self.accelerator.is_main_process:
                log_dict = self.eval(epoch=-1)
                log_dict['epoch'] = -1
                wandb.log(log_dict)
            self.accelerator.wait_for_everyone()
        
        # II. Training loop
        with tqdm.tqdm(total=self.t_args.optim.n_epochs * len(self.dataloader_consistency), disable=not self.accelerator.is_main_process) as pbar:
            for epoch in range(self.t_args.optim.n_epochs):
                pbar.set_description("/!\ DEBUGING /!\ " if self.debug else f"   epoch [{epoch}/{self.t_args.optim.n_epochs}]")
                train_epoch_start = time.time()

                # 1. Freeze tinymlp
                if epoch == self.t_args.optim.freeze_tinymlp_after_n_epoch:
                    self.optimizers['latent_nerfs'].param_groups[0]['lr'] = 0

                # 2. Iterate over the dataloader and update NerfVAE
                for iter_nb, batch_consistency in enumerate(self.dataloader_consistency):
                    do_injection = self.t_args.injection.apply and do_now(iter_nb, self.t_args.injection.every)
                    
                    if not do_injection:
                        self.consistency_iter(batch_consistency)

                    if do_injection :
                        batch_injection = next(self.iterator_injection)

                        self.consistency_and_injection_iter(
                            batch_consistency,
                            batch_injection,
                            strategy=self.t_args.injection.strategy
                        )

                    # # for debuging
                    # if self.nerf_vae.latent_nerfs.local_planes.grad.isnan().sum() > 0:
                    #     print("WARNING: latent_nerf grad has nan values")

                    if self.accelerator.is_main_process:
                        pbar.update(1)
                train_epoch_end = time.time()
                self.cum_train_time += train_epoch_end - train_epoch_start

                # 3.Scheduler step at the end of the epoch
                self.step_relevant_schedulers()

                # $. Evaluation, logging and saving
                if self.accelerator.is_main_process:
                    if self.config.wandb.apply:
                        # Evaluation
                        eval_epoch_start = time.time()
                        log_dict = self.eval(
                            epoch,
                            consistency_metrics=do_now(epoch, self.t_args.logging.metrics_every_epoch),
                            injection_metrics=do_now(epoch, self.t_args.logging.injection_metrics_every_epoch),
                            consistency_dashboard=do_now(epoch, self.t_args.logging.consistency_dashboard_every_epoch),
                            injection_dashboard=do_now(epoch, self.t_args.logging.injection_dashboard_every_epoch),
                            nerfs_video=do_now(epoch, self.t_args.logging.eval_video_every_epoch),
                        )
                        eval_epoch_end = time.time()
                        self.cum_eval_time += eval_epoch_end - eval_epoch_start

                        # Train metrics
                        if do_now(epoch, self.t_args.logging.log_training_losses_every_epoch) :
                            log_dict['train_losses/'] = self.get_log_dict_from_meters()
                            self.reset_meters()

                    # Saving
                    if (epoch == self.t_args.optim.n_epochs - 1) or do_now(epoch, self.t_args.logging.save_every_epoch):
                        self.save(epoch)

                    if self.config.wandb.apply:
                        # Logging
                        if log_dict:
                            log_dict['epoch'] = epoch
                            log_dict['cum_train_time'] = self.cum_train_time
                            log_dict['cum_eval_time'] = self.cum_eval_time
                            wandb.log(log_dict)

                self.accelerator.wait_for_everyone()

        # III. Training is finished
        if self.accelerator.is_main_process and self.config.wandb.apply:
            wandb.finish()

    def eval(
            self, 
            epoch, 
            consistency_dashboard=True,
            injection_dashboard=True,
            nerfs_video=True,
            injection_metrics=True,
            consistency_metrics=True,
            ):
        "Return a dictionnary with the requested evaluations"
        
        # unwrap models
        if any([consistency_dashboard, injection_dashboard, nerfs_video, injection_metrics, consistency_metrics]):
            _nerf_vae = self.accelerator.unwrap_model(self.nerf_vae)
            
        # Perform evaluations
        eval_dict = {}
        if consistency_dashboard:
            eval_dict.update(
                self.evaluator.get_consistency_dashboard(
                    epoch,
                    _nerf_vae.vae,
                    _nerf_vae.latent_nerfs,
                    _nerf_vae.latent_renderer,
                    self.normalizer,
                    batch_size=self.t_args.consistency.batch_size,
                    device=self.accelerator.device)
                )

        if injection_dashboard:
            eval_dict.update(
                self.evaluator.get_injection_dashboard(
                    epoch, 
                    _nerf_vae.vae, 
                    self.normalizer,
                    batch_size=max(self.t_args.injection.batch_size, self.t_args.consistency.batch_size),
                    device=self.accelerator.device
                )
            )   

        if nerfs_video:
            eval_dict['nerfs_video/'] = self.evaluator.get_video_eval(
                _nerf_vae.vae, 
                _nerf_vae.latent_nerfs[:self.config.eval.video.n_scenes], 
                _nerf_vae.latent_renderer, 
                self.normalizer, 
                batch_size=self.t_args.consistency.batch_size, 
                device=self.accelerator.device
            )
            if self.config.global_latent_nerf.apply and (self.config.latent_nerf.n_local_features == 0 or self.config.global_latent_nerf.fusion_mode == 'sum'):
                # In this case, the global latent nerfs can be visualized using the same tinyMLP as the scene nerfs
                eval_dict['global_nerfs_video/'] = self.evaluator.get_video_eval(
                    _nerf_vae.vae, 
                    _nerf_vae.latent_nerfs.global_planes, 
                    _nerf_vae.latent_renderer, 
                    self.normalizer, 
                    batch_size=self.t_args.consistency.batch_size, 
                    device=self.accelerator.device
                )

        if injection_metrics:
            eval_dict['injection_metrics/'] = self.evaluator.get_injection_metrics(
                _nerf_vae.vae,
                batch_size=max(self.t_args.injection.batch_size, self.t_args.consistency.batch_size),
                device=self.accelerator.device
            )

        if consistency_metrics:
            n_scenes = self.config.eval.metrics.n_scenes
            if n_scenes == 'max':
                slice_= slice(None)
            else:
                slice_ = slice(0, n_scenes)
            eval_dict['consistency_metrics/'] = self.evaluator.get_consistency_metrics(
                _nerf_vae.vae,
                _nerf_vae.latent_nerfs[slice_],
                _nerf_vae.latent_renderer,
                self.normalizer,
                batch_size=self.t_args.consistency.batch_size,
                device=self.accelerator.device
            )

        return eval_dict

    def save(self, epoch):
        "Save the models and the config at current state"
        savedir = os.path.join(self.repo_path, self.config.savedir, self.expname)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        save_config(self.config, "config.yaml", savedir)

        _nerf_vae = self.accelerator.unwrap_model(self.nerf_vae)
        save_models_args = {
            'savedir' : savedir,
            'vae' : _nerf_vae.vae, 
            'nerfs' : _nerf_vae.latent_nerfs, 
            'renderer' : _nerf_vae.latent_renderer, 
            'include_global_nerfs' : self.config.global_latent_nerf.apply,
        }
        save_models("gvae_latest.pt", **save_models_args)
        if not self.t_args.logging.save_latest_only :
            save_models(f"gvae_epoch_{epoch}.pt", save_models_args)


def init_datasets(config) : 
    data_reader = DataReader(config.dataset)
    train_injection_dataset = ImageDataset(config.injection_dataset, split='train') 
    test_injection_dataset = ImageDataset(config.injection_dataset, split='test')

    pose_sampler = LazyPoseSampler(
        dataset_name=config.dataset.name,
    )
    num_scenes = len(data_reader.path_of_scenes)
    train_scenes = [ 
        ObjectDataset(data_reader, selected_obj_idx=i, split='train')
        for i in range(num_scenes)
    ]
    multi_scene_trainset = MultiSceneDataset(train_scenes)
    test_scenes = [ 
        ObjectDataset(data_reader, selected_obj_idx=i, split='test')
        for i in range(num_scenes)
    ]
    multi_scene_testset = MultiSceneDataset(test_scenes)
    return train_scenes, multi_scene_trainset, train_injection_dataset, test_scenes, multi_scene_testset, test_injection_dataset, pose_sampler, num_scenes

def init_models(config, n_scenes) : 

    if config.rgb_override:
        return init_models_rgb_override(config, n_scenes)
    
    # 1. Vae
    baseline_vae_manager = BaselineVaeManager(config.vae.pretrained_model_name_or_path)
    vae = baseline_vae_manager.get_vae()

    config.latent_nerf.rendering_options['bg_color'] = baseline_vae_manager.get_bg_color()

    n_features = config.latent_nerf.n_local_features
    if config.global_latent_nerf.apply:
        if config.global_latent_nerf.fusion_mode == "concat":
            n_features = config.global_latent_nerf.n_global_features + config.global_latent_nerf.n_local_features
        elif config.global_latent_nerf.fusion_mode == "sum":
            n_features = config.global_latent_nerf.n_global_features

    #2. Renderer
    rendering_resolution = config.dataset.img_size / baseline_vae_manager.get_downscale_factor()
    assert rendering_resolution.is_integer(), "Rendering resolution should be an integer"
    latent_renderer = TriPlaneRenderer(
        neural_rendering_resolution=int(rendering_resolution),
        n_channels=baseline_vae_manager.get_num_latent_channels(),
        n_features=n_features,
        aggregation_mode=config.latent_nerf.aggregation_mode,
        rendering_kwargs=config.latent_nerf.rendering_options,
        use_coordinates=False,
        use_directions=False
    )

    #3. Nerfs
    latent_nerfs =  TriplaneManager(
            n_scenes=n_scenes,
            local_nerf_cfg=config.latent_nerf,
            global_nerf_cfg=config.global_latent_nerf,
    )
    
    return vae, latent_renderer, latent_nerfs

def init_models_rgb_override(config, n_scenes) :

    class EncoderOutput:
        def __init__(self, mean):
            self.mean = mean
            self.latent_dist = IdentityDistribution(mean)

    class DecoderOutput():
        def __init__(self, sample):
            self.sample = sample

    class IdentityDistribution():
        def __init__(self, x):
            self.mean = x
            self.std = torch.zeros_like(x)

        def sample(self):
            return self.mean

    class IdentityEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return EncoderOutput(x)
    
    class IdentityDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return DecoderOutput(x)

    class IdentityVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = IdentityEncoder()
            self.decoder = IdentityDecoder()

        def encode(self, x):
            return self.encoder(x)
        
        def decode(self, z):
            return self.decoder(z)
        
        def forward(self, x):
            return self.decoder(self.encoder(x))
    
    vae = IdentityVAE()

    n_features = config.rgb_nerf_as_latent_override.n_local_features
    if config.global_latent_nerf.apply:
        if config.global_latent_nerf.fusion_mode == "concat":
            n_features = config.global_latent_nerf.n_global_features + config.global_latent_nerf.n_local_features
        elif config.global_latent_nerf.fusion_mode == "sum":
            n_features = config.global_latent_nerf.n_global_features

    latent_renderer = TriPlaneRenderer(
        neural_rendering_resolution=config.dataset.img_size,
        n_channels=3,
        n_features=n_features,
        aggregation_mode=config.rgb_nerf_as_latent_override.aggregation_mode,
        rendering_kwargs=config.rgb_nerf_as_latent_override.rendering_options,
        use_coordinates=False,
        use_directions=False
    )
    latent_nerfs =  TriplaneManager(
            n_scenes=n_scenes,
            local_nerf_cfg=config.rgb_nerf_as_latent_override,
            global_nerf_cfg=config.global_latent_nerf,
    )
    return vae, latent_renderer, latent_nerfs
