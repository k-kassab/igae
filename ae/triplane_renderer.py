# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch

from ae.volume_rendering.renderer import ImportanceRenderer
from ae.volume_rendering.ray_sampler import RaySampler


class TriPlaneRenderer(torch.nn.Module):



    def __init__(self,                   
        neural_rendering_resolution,             # Output resolution.
        n_channels,
        n_features,
        aggregation_mode,
        use_coordinates,
        use_directions,
        positionnal_encoding_kwargs={},
        rendering_kwargs={},
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_features = n_features
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.decoder = OSGDecoder(n_features, self.n_channels, aggregation_mode, use_coordinates, use_directions, positionnal_encoding_kwargs)
        self.neural_rendering_resolution = neural_rendering_resolution
        self.rendering_kwargs = rendering_kwargs
    

    def synthesis(self, planes, pose, neural_rendering_resolution=None):
        bs, n_renders = pose.shape[:2]
        cam2world_matrix = pose[..., :16].view(bs*n_renders, 4, 4)
        intrinsics = pose[..., 16:25].view(bs*n_renders, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
        ray_origins = ray_origins.reshape(bs, n_renders*neural_rendering_resolution**2, 3)
        ray_directions = ray_directions.reshape(bs, n_renders*neural_rendering_resolution**2, 3)
        planes = planes.view(len(planes), 3, self.n_features, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.reshape(bs, n_renders, H, W, feature_samples.shape[-1]).permute(0,1,4,2,3)# /!\ permute affected
        depth_image = depth_samples.reshape(bs, n_renders, H, W, 1).permute(0,1,4,2,3) # /!\ permute affected
        # the final shape for feature_image and depth_image here is : [bs, n_renders, C, H, W]
        
        res = {
            'img': feature_image[:, :, :self.n_channels], 
            'img_depth': depth_image
        }  

        return res 
    

    def forward(self, planes, pose, neural_rendering_resolution=None, aggregation_mode='sum'):
        """ Given a batch of scenes (i.e. `triplanes`), performs then returns n renderings per scene, n being 
        the number of pose given per scene.
        args:
            planes: torch.Tensor: Triplane representation with shape [bs, 3, n_features, plane_resolution, plane_resolution] or
                    [bs, 3 * n_features, plane_resolution, plane_resolution, plane_resolution]
            pose: torch.Tensor: Camera pose (extrinsics+intrinsics) with shape [bs, n_renders, 25]
            neural_rendering_resolution: int: Resolution of the neural-rendered images.

        returns:
            img: torch.Tensor: Renderings with values in [-1, 1] and shape [bs, n_renders, n_channels, neural_rendering_resolution, neural_rendering_resolution]
            img_depth: torch.Tensor: Depth map with values in [ray_start, ray_end] and shape [bs, n_renders, 1, neural_rendering_resolution, neural_rendering_resolution] 
        """
        return self.synthesis(planes, pose, neural_rendering_resolution=neural_rendering_resolution)



class Embedder:
    def __init__(self, mode:str, num_freqs:int):
        self.mode = mode
        self.embed_kwargs = {
            'include_input' : True,
            'num_freqs' : num_freqs,
            'max_freq_log2' : num_freqs-1,
            'log_sampling' : True,
            'periodic_fns' : [torch.sin, torch.cos],
        }

        self.emb = self.create_embedding_fn()
    
    def calculate_emb_dim(self, input_dim):
        if self.mode == 'on' : 
            out_dim = 0 
            if self.embed_kwargs['include_input']:
                out_dim += input_dim
            
            out_dim += input_dim * self.embed_kwargs['num_freqs'] * len(self.embed_kwargs['periodic_fns'])
            return out_dim
        
        elif self.mode == 'off' :
            return input_dim

    def create_embedding_fn(self):
        embed_fns = []

        if self.embed_kwargs['include_input']:
            embed_fns.append(lambda x : x)
            
        max_freq = self.embed_kwargs['max_freq_log2']
        N_freqs = self.embed_kwargs['num_freqs']
        
        if self.embed_kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.embed_kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
        
        if self.mode == 'on' : 
            return lambda x: torch.cat([fn(x) for fn in embed_fns], -1)
        elif self.mode == 'off' :
            return torch.nn.Identity()
        else:
            raise NotImplementedError(f"Positionnal encoding mode can be either 'on' or 'off'. {self.mode} was given")

    def __call__(self, inputs):
        return self.emb(inputs)


class OSGDecoder(torch.nn.Module):
    def __init__(
            self, 
            n_features_in:int, 
            n_channels_out:int, 
            aggregation_mode:str,
            use_coordinates:bool=False,
            use_directions:bool=False,
            positionnal_encoding_kwargs={}
        ): 
        """
        Tiny MLP that decodes the 3 features obtained from the triplame in an color and a density.
        args:
            n_features_in: Dimension of the features of the triplane.
            n_channels_out: Number of channels of the output image.
            aggregation_mode: How to aggregate the features of the triplane. Can be 'prod' or 'sum'.
            use_coordinates: Whether to include the coordinates of the point in the decoder.
            use_directions: Whether to include the directions of the point in the decoder.

        """
        super().__init__()
        self.hidden_dim = 64
        self.aggregation_mode = aggregation_mode
        self.use_coordinates = use_coordinates
        self.use_directions = use_directions

        if use_coordinates or use_directions:
            self.embedder = Embedder(**positionnal_encoding_kwargs)

        input_dim = n_features_in 
        if self.use_coordinates:
            input_dim += self.embedder.calculate_emb_dim(3)
        if self.use_directions:
            input_dim += self.embedder.calculate_emb_dim(3)
            
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.hidden_dim),
            torch.nn.Softplus(),
            torch.nn.Linear(self.hidden_dim, 1 + n_channels_out)
        )    

    def forward(self, features, coordinates, directions):
        """
        features: tensor w shape [bs, 3, n_samples, feature_dim]
        coordinates: tensor w shape [bs, n_samples, 3]
        directions: tensor w shape [bs, n_samples, 3]
        """
        # Aggregate features
        if self.aggregation_mode=='sum':
            features = features.sum(1)
        elif self.aggregation_mode=='prod':
            features = features.prod(1)
        else:
            raise NotImplementedError(f"Aggregation mode '{self.aggregation_mode}' not supported.")

        # building the input of the timy mlp
        x = features
        if self.use_coordinates:
            x = torch.cat([x, self.embedder(coordinates)], dim=-1)
        if self.use_directions:
            x = torch.cat([x, self.embedder(directions)], dim=-1)

        # forwarding through the tiny mlp
        x = self.net(x)

        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}

