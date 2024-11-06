# Copyright 2024 Antoine Schnepf, Karim Kassab, Jean-Yves Franceschi, Laurent Caraffa, 
# Flavian Vasile, Jeremie Mary, Andrew Comport, Valérie Gouet-Brunet

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch 
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import imageio
import tqdm
import os
import math
import yaml
import sys
import collections.abc
from prodict import Prodict

# ----- video ------ #
render_all_default_options = {'include_latents': True, 'separate_latents':True, 'include_depth':True, 'contrast_scale':1, 'rgb_override': False}

@torch.no_grad()
def make_dashboard_video(triplane_renderer, triplanes, pose_sampler, latent_to_pil_transform, 
                         latent_to_rgb_transform, rendering_options, render_batch_size, device, save_name, save_dir, 
                         fps=30,n_frames=60, azimuth_range=(0,1), elevation_range=(0.3,0.3), radius_range=(1.3, 1.3), quiet=False, options=render_all_default_options) : 
    """
    creates a video for the current triplane
    args:
        `dashboard_shape`: tuple of ints (`n_rows`, `n_cols`) where `n_rows` is the number of set of input images, 
            and `n_cols` is the number of noise levels that have been tested. 
        `source_image`: stacked inputs image with shape [n_rows, n_multiview_images, 3, img_size, img_size]
        `triplanes`: stacked triplanes with shape [n_rows * n_noise_levels, 3*feature_dim, img_size, img_size]
        `azimuth_range`: (0,1) means a full 360
        `elevation` range: (-1,1) means from completely above to completely below
    """
    n_videos = len(triplanes)

    # Generate the videos frames
    poses = gen_video_poses(n_frames, pose_sampler, azimuth_range, elevation_range, radius_range, n_triplanes=len(triplanes))
    video_out = imageio.get_writer(os.path.join(save_dir, save_name), mode='I', fps=fps, codec='libx264')
    local_img_stack = []
    for i, (imgs_dict) in tqdm.tqdm(
                enumerate(render_all_poses_via_emulated_batches(triplane_renderer, poses, triplanes, render_batch_size, latent_to_pil_transform, latent_to_rgb_transform, rendering_options, device=device, mode='transpose', options=options)),      
                total=n_videos*n_frames, 
                desc="Creating video dashboard", disable=quiet
        ) : 

        rgb_img = imgs_dict['rgb_img']

        # resize latent img and depth img to match rgb_img
        for key in imgs_dict.keys() : 
                imgs_dict[key] = imgs_dict[key].resize(rgb_img.size)

        stack = list(imgs_dict.values())
        local_img_stack.append(image_grid(stack, len(stack), 1))

        if i % n_videos == n_videos - 1 :
            frame = image_grid(local_img_stack, rows=1, cols=n_videos)
            video_out.append_data(np.array(frame))
            local_img_stack = []
    video_out.close()

@torch.no_grad()
def render_all_poses_via_emulated_batches(triplane_renderer, poses, triplane, render_batch_size, latent_to_pil_transform, 
                                          latent_to_rgb_transform, rendering_options, device, mode='natural', 
                                          options=render_all_default_options) : 
    """Generator object for rendering many triplane under many camera poses.
    Expected args : 
        poses: torch.tenso with shape [n_triplanes, n_renders, 25]
        triplane: torch.tensor with shape [n_triplanes, 3, feature_dim, triplane_resolution, triplane_resolution]
        render_batch_size: max number of images to be rendered in a single forward pass. 
    Generate PIL images. The order of generation can be controlled by the mode argument.
    """

    n_triplanes, n_renders = poses.shape[:2]

    if mode == 'natural' :
        # maximize the number of renders per batch in the limit of render_batch_size
        if n_renders >= render_batch_size : 
            n_triplanes_per_batch = 1
            n_pose_per_batch = render_batch_size
        else : 
            n_triplanes_per_batch = math.floor(render_batch_size / n_renders)
            n_pose_per_batch = math.floor(render_batch_size / n_triplanes_per_batch)

        def outter_loop():
            for idx_tri in range(math.ceil(n_triplanes / n_triplanes_per_batch)) :
                for idx_pose in range(math.ceil(n_renders / n_pose_per_batch)) : 
                    yield idx_tri, idx_pose

        def inner_loop(n_tri, n_ren):
            for iidx_tri in range(n_tri) : 
                for iidx_pose in range(n_ren) : 
                    yield iidx_tri, iidx_pose

    elif mode == 'transpose' :
        # maximizee the number of triplane per batch. Set renders_per_batch accordingly
        if n_triplanes >= render_batch_size :
            n_pose_per_batch = 1
            n_triplanes_per_batch = render_batch_size
        else : 
            n_pose_per_batch = math.floor(render_batch_size / n_triplanes)
            n_triplanes_per_batch = math.floor(render_batch_size / n_pose_per_batch)

        def outter_loop():
            for idx_pose in range(math.ceil(n_renders / n_pose_per_batch)) : 
                for idx_tri in range(math.ceil(n_triplanes / n_triplanes_per_batch)) :
                    yield idx_tri, idx_pose

        def inner_loop(n_tri, n_ren):
            for iidx_pose in range(n_ren) : 
                for iidx_tri in range(n_tri) : 
                    yield iidx_tri, iidx_pose

    else :
        raise ValueError(f"mode {mode} not recognized")
    
    for idx_tri, idx_pose in outter_loop() :
        batched_pose = poses[idx_tri*n_triplanes_per_batch : (idx_tri+1)*n_triplanes_per_batch, idx_pose*n_pose_per_batch : (idx_pose+1)*n_pose_per_batch].to(device)
        batched_triplane = triplane[idx_tri*n_triplanes_per_batch : (idx_tri+1)*n_triplanes_per_batch].to(device)
        out = triplane_renderer(batched_triplane, batched_pose)

        n_tri, n_ren = batched_pose.shape[:2]
        for iidx_tri, iidx_pose in inner_loop(n_tri, n_ren) : 
                imgs_dict = {}
                raw_latent_img = out['img'][iidx_tri, iidx_pose]

                if options['include_latents']:
                    if options['separate_latents'] : 
                        for c in range(raw_latent_img.shape[-3]) : 
                            imgs_dict[f'latent_img_channel_{c}'] = latent_to_pil_transform(raw_latent_img[c:c+1])
                    else: 
                        limg = latent_to_pil_transform(raw_latent_img[:3])
                        limg = ImageEnhance.Contrast(limg).enhance(options['contrast_scale'])
                        imgs_dict['latent_img'] = limg

                imgs_dict['rgb_img'] = latent_to_rgb_transform(raw_latent_img)

                if options['include_depth'] :
                    imgs_dict['depth_img'] = color_depth_map(
                        out['img_depth'][iidx_tri, iidx_pose, 0].cpu().numpy(),
                        rs=rendering_options['ray_start'],
                        re=rendering_options['ray_end']
                    )
                
                yield imgs_dict

@torch.no_grad()
def gen_video_poses(n_frames, pose_sampler, azimuth_range, elevation_range, radius_range, n_triplanes=1) : 
    poses = pose_sampler.get_pose(
        azimuth=torch.linspace(azimuth_range[0], azimuth_range[1], n_frames)*3.14*2,
        elevation=torch.linspace(elevation_range[0], elevation_range[1], n_frames)*3.14/2,
        radius=torch.linspace(radius_range[0], radius_range[1], n_frames)
    )
    return poses.unsqueeze(0).repeat_interleave(n_triplanes, dim=0)

def color_depth_map(imgarray, rs, re, cm=plt.get_cmap('plasma')) : 
    "imgarray : array of shape (H, W) with value in [rs, re]"
    cm_img = cm((imgarray - rs)/(re-rs))
    return Image.fromarray((cm_img[..., :3] * 255).astype(np.uint8))

def image_grid(imgs, rows, cols) : 
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i in range(rows):
        for j in range(cols) : 
            grid.paste(imgs[cols*i +j], box=(j*w, i*h))

    return grid


# ----- config ------ #
class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        assert key in source.keys(), f"key {key} not in source"
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]

    return source

def yaml_load(cfg_name, load_dir):
    config_path = os.path.join(load_dir, cfg_name)
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def load_config(cfg_name, load_dir, from_default=False, default_cfg_name='default.yaml') :
    """Load a configuration file. If from_default is True, load 
    the default config and update it with the config file"""
    
    config = yaml_load(cfg_name, load_dir)

    if from_default :
        default_config = yaml_load(default_cfg_name, load_dir)
        config = deep_update(default_config, config)

    return config

def save_config(config, cfg_name, save_dir) :
    config_path = os.path.join(save_dir, cfg_name)
    if isinstance(config, Prodict) :
        config = Prodict.to_dict(config, is_recursive=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def print_dict(dico):
    for u,v in dico.items():
        print(f"{u} : {v}")


# ----- loss ------ #
def compute_tv(t:torch.Tensor, p=2, q=2):
    "Computes TV ||.||_p^q over the last two dimensions"
    assert p in [1, 2] and q in [1, 2], f"p and q must be in [1, 2] but are {p} and {q}"
    positify = lambda x : torch.norm(x, p=p)**q

    h_tv = positify(t[..., 1:, :] - t[..., :-1, :]).sum()
    w_tv = positify(t[..., :, 1:] - t[..., :, :-1]).sum()
    
    return (h_tv + w_tv) / (t.shape[-2] + t.shape[-1])

# def compute_tv(t:torch.Tensor, p=2, q=2):
#     "Computes TV ||.||_p^q over the last two dimensions"
#     assert p in [1, 2] and q in [1, 2], f"p and q must be in [1, 2] but are {p} and {q}"
#     positify = lambda x : torch.norm(x, p=p)**q

#     h_tv = positify(t[..., 1:, :] - t[..., :-1, :]).sum()
#     w_tv = positify(t[..., :, 1:] - t[..., :, :-1]).sum()
    
#     return (h_tv + w_tv) / (t.shape[-2] + t.shape[-1])

def set_requires_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def remove_empty_categories(config):
    clean_config = Prodict.from_dict(config.copy())
    for cat_name, cat_range in config.dataset.scene_repartition.items():
        if cat_range[0] == 0 and cat_range[1] == 0:
            clean_config.dataset.scene_repartition.pop(cat_name)
    return clean_config

# ----- other ------ #
def do_now(i, every) :
    return i % every == every - 1

def to_dict(my_tensor) : 
    return {
        f"channel_{k}" : my_tensor[..., k].item()
        for k in range(my_tensor.shape[-1])
    }