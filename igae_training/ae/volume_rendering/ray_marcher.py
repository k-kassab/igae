# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The ray marcher takes the raw output of the implicit representation and uses the volume rendering equation to produce composited colors and depths.
Based off of the implementation in MipNeRF (this one doesn't do any cone tracing though!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MipRayMarcher2(nn.Module):
    def __init__(self):
        super().__init__()


    def run_forward(self, colors, densities, depths, rendering_options):
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2


        if rendering_options['clamp_mode'] == 'softplus':
            densities_mid = F.softplus(densities_mid - 1) # activation bias of -1 makes things initialize better
        elif rendering_options['clamp_mode'] == 'relu':
            densities_mid = F.relu(densities_mid)
        else:
            assert False, "MipRayMarcher only supports `clamp_mode`=`softplus`!"

        density_delta = densities_mid * deltas

        alpha = 1 - torch.exp(-density_delta)

        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]

        composite_rgb = torch.sum(weights * colors_mid, -2)
        weight_total = weights.sum(2)
        composite_depth = torch.sum(weights * depths_mid, -2) / weight_total

        # clip the composite to min/max range of depths
        composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        composite_depth = torch.clamp(composite_depth, torch.min(depths), torch.max(depths))

        if rendering_options.get('bg_color', False):
            bg_color = torch.ones_like(composite_rgb)
            bg_color[...,:] = torch.tensor(rendering_options['bg_color'])
            composite_rgb = composite_rgb + (1 - weight_total) * bg_color # white

        composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)

        return composite_rgb, composite_depth, weights


    def forward(self, colors, densities, depths, rendering_options):
        composite_rgb, composite_depth, weights = self.run_forward(colors, densities, depths, rendering_options)

        return composite_rgb, composite_depth, weights
    

if __name__ == '__main__':  
    # inspecting how we can backpropagate into the densities in volume rendering
    import numpy as np
    import matplotlib.pyplot as plt

    def unnormalize(color) :
        "from [-1, 1] to [0, 1]"
        return (color + 1) / 2
    
    def normalize(color):
        "from [0, 1] to [-1, 1]"
        return color * 2 - 1
    
    rendering_options = {
    'clamp_mode' : 'softplus', #'relu',
    'bg_color': [1.,1.,1.], 
} 
    
    volume_renderer = MipRayMarcher2()
 
    with torch.no_grad() : 
        depths = torch.linspace(0, 1, 64).unsqueeze(0).unsqueeze(0).unsqueeze(-1) # shape [1 , 1, 64, 1] i.e batch_size (n_img), n_pixel (img_size**2), n_sample_along_ray, 1
        densities = torch.abs(torch.sin(depths * 5)) * 3 # shape [1, ,1 , 64, 1]
        densities = torch.randn_like(depths)
        colors = torch.tensor( 
            [[1, u/2, v/2] for u,v in zip(1 - np.sin(np.linspace(-2, 4*np.pi, 64)), 1 - np.cos(np.linspace(0, 2*np.pi, 64)))]
        ).unsqueeze(0).unsqueeze(0).clamp(0, 1)
        colors = normalize(colors).float()

    x = depths[0,0, :, 0].detach().numpy()
    y0 = densities[0, 0, :, 0].detach().numpy()


    gt_color = normalize(torch.Tensor([1,1,1]).float())
    eps = 100
    n_iter = 500
    losses = []
    for i in range(500) : 
        
        densities.requires_grad = True
        composite_rgb, composite_depth, weights = volume_renderer(colors, densities, depths, rendering_options)
        if i == 0 : 
            plt.imshow(unnormalize(composite_rgb.detach().numpy()))
            plt.title(f"INITIAL RENDERING. depths:{composite_depth.item():.2f}. Cumulated weight: {weights.sum(2).item():.2f}")
            plt.show()

        loss = F.mse_loss(composite_rgb.squeeze(0).squeeze(0), gt_color)
        losses.append(loss.item())
        loss.backward()
        with torch.no_grad():
            densities = densities - densities.grad * eps
        

    plt.plot(losses)
    plt.title('losses')
    plt.show()

    x = depths[0,0, :, 0].detach().numpy()
    y = densities[0, 0, :, 0].detach().numpy()
    plt.title('density vs depth')
    plt.plot(x,y, label="final density")
    plt.plot(x,y0, label="initial density")
    plt.plot(x, np.zeros_like(x), '--', color='red')
    plt.legend()
    plt.show()

    plt.plot(x, list(colors[0,0]))
    plt.title('colors vs depth')
    plt.show()

    plt.imshow(unnormalize(composite_rgb).detach().numpy())
    plt.title(f"PRED. depths:{composite_depth.item():.2f}. Cumulated weight: {weights.sum(2).item():.2f}")
    plt.show()
    plt.imshow(unnormalize(gt_color).unsqueeze(0).unsqueeze(0).detach().numpy())
    plt.title(f"GT. depths:{composite_depth.item():.2f}")
    plt.show()