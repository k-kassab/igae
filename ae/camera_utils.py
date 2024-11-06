# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# This file has been modified to be used in the context of the IG-AE project.

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Union
from ae.volume_rendering import math_utils
from datasets.dataset import DatasetIntrinsicsManager

@torch.no_grad()
def retrieve_azi_elev_radius_from_extrinsics(extrinsics, return_azimuth_sines=False) : 
    """Given a batch of extrinsics, returns the corresponding batch of azimuth, elevation and radius."""
    radius = torch.norm(extrinsics[..., :3, 3], dim=-1)
    position_on_unit_sphere = extrinsics[..., :3, 3] / radius[..., None]
    almost_elevation = torch.acos(position_on_unit_sphere[..., 2])
    elevation = torch.pi/2 - almost_elevation
    position_on_unit_circle = position_on_unit_sphere[..., :2] / torch.sin(almost_elevation)[..., None]
    azimuth_cos = position_on_unit_circle[..., 0]
    azimuth_sin = position_on_unit_circle[..., 1]

    if return_azimuth_sines : 
        return azimuth_cos, azimuth_sin, elevation, radius
    
    azimuth = torch.atan2(azimuth_sin, azimuth_cos)
    return azimuth, elevation, radius


def retrieve_azi_elev_radius_from_pose(pose, return_azimuth_sines=False) : 
    """Given a batch of poses, returns the corresponding batch of azimuth, elevation and radius."""
    extrinsics = pose[..., :16]
    extrinsics = extrinsics.view(*extrinsics.shape[:-1], 4, 4)

    return retrieve_azi_elev_radius_from_extrinsics(extrinsics, return_azimuth_sines=return_azimuth_sines)


def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes z-axis is up and that there is no camera roll.
    """

    forward_vector = math_utils.normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 0, 1], dtype=torch.float, device=origin.device).expand_as(forward_vector)

    right_vector = -math_utils.normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = math_utils.normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world


class PoseSampler: 
    
    @staticmethod
    def get_intrinsics(focal, cx=0.5, cy=0.5, batch_size=1, device='cpu') : 
        """
        Returns the intrinsics 3x3 matrix given azimuth, elevation and radius. 
        """
        intrinsics = torch.zeros((3,3), device=device).float()
        intrinsics[0,0] = focal
        intrinsics[1,1] = focal
        intrinsics[0,2] = cx
        intrinsics[1,2] = cy
        return intrinsics.unsqueeze(0).repeat_interleave(batch_size, dim=0) 
    
    @staticmethod
    def _get_extrinsics_from_batched_elev_azi(elevations, azimuths, radius) : 
        """return a cam2world matrix from elevations and azimuths angles. Angles are expected in radians.
        azimuths: torch.tensor of shape [batch_size, 1] with each scalar value expected in [-pi, pi]
        elevations: torch.tensor of shape [batch_size, 1] with each scalar value expected in ]-pi/2, pi/2[
        """
        batch_size = elevations.shape[0]
        device=elevations.device

        camera_origins = torch.zeros((batch_size, 3), device=device)
        camera_origins[:, 0] = radius*torch.cos(elevations) * torch.cos(azimuths)
        camera_origins[:, 1] = radius*torch.cos(elevations) * torch.sin(azimuths)
        camera_origins[:, 2] = radius*torch.sin(elevations)

        return create_cam2world_matrix(-camera_origins, camera_origins)

    @staticmethod
    def get_extrinsics(azimuth:Union[torch.Tensor, float], elevation:Union[torch.Tensor, float], radius:Union[torch.Tensor, float], batch_size=None, device='cpu', return_batch_size=False) : 
        """
        Returns the extrinsics 4x4 matrix given azimuth, elevation and radius. 
        Each of the argument can be a scalar or a torch.Tensor of shape (batch_size, ).
        If only scalars are given, make sure to specify the batch_size. If one of the arguments is a torch.Tensor, the batch_size is inferred from it. 
        If several tensors are given, make sure they must have the same batch_size.
        """

        # 1. Infer desired batch_size
        if isinstance(azimuth, torch.Tensor) :
            batch_size = azimuth.shape[0]
        elif isinstance(elevation, torch.Tensor) :
            batch_size = elevation.shape[0]
        elif isinstance(radius, torch.Tensor) :
            batch_size = radius.shape[0]
        else :
            assert batch_size, "If all arguments are scalars, batch_size must be specified."
        
        # 2. Convert scalars to tensors
        if isinstance(azimuth, Union[float, int]) : 
            azimuth = torch.zeros((batch_size, ), device=device) + azimuth
        if isinstance(elevation, Union[float, int]) :
            elevation = torch.zeros((batch_size, ), device=device) + elevation
        if isinstance(radius, Union[float, int]) :
            radius = torch.zeros((batch_size, ), device=device) + radius
        
        assert elevation.shape[0] == azimuth.shape[0] == radius.shape[0], "All non-scalar arguments must have the same batch_size."

        #3. Compute extrinsics
        extrinsics = PoseSampler._get_extrinsics_from_batched_elev_azi(elevation, azimuth, radius)

        if return_batch_size:
            return extrinsics, batch_size
        
        return extrinsics

    @staticmethod
    def get_pose(focal:float, azimuth:Union[torch.Tensor, float], elevation:Union[torch.Tensor, float], radius:Union[torch.Tensor, float], cx:float=0.5, cy:float=0.5, batch_size=None, device='cpu'):
        """
        Returns the 25-dimentional pose vector given 
            -focal, focal displacement cx and cy (instrinsics)
            -azimuth, elevation and radius (=extrinsics)
        Each of the extrinsics argument can be a scalar or a torch.Tensor of shape (batch_size, ).
        If only scalars are given, make sure to specify the batch_size. If one of the arguments is a torch.Tensor, the batch_size is inferred from it. 
        If several tensors are given, make sure they must have the same batch_size.
        """

        extrinsics, batch_size = PoseSampler.get_extrinsics(azimuth, elevation, radius, batch_size, device, return_batch_size=True)
        intrinsics = PoseSampler.get_intrinsics(focal, cx, cy, batch_size, device)
        return torch.cat([extrinsics.reshape(batch_size, 16), intrinsics.reshape(batch_size, 9)], dim=1)

    @staticmethod
    def sample_extrinsics_gaussian(azimuth_mean, elevation_mean, azimuth_std=0, elevation_std=0, radius=1, batch_size=1, device='cpu'):
        """
        Samples points on a sphere from a Gaussian distribution and returns a camera pose.
        Camera is specified as looking at the origin.
        If horizontal and vertical stddev (specified in radians) are zero, gives a
        deterministic camera pose.
        The angles are expected in radians.
        """

        azimuths = torch.randn((batch_size, ), device=device) * azimuth_std + azimuth_mean
        elevations = torch.randn((batch_size, ), device=device) * elevation_std + elevation_mean

        # correct the oversampling of the poles
        elevations = torch.clamp(elevations, -torch.pi/2 + 1e-5, torch.pi/2 - 1e-5)
        elevations = torch.arccos(- 2 * elevations/ torch.pi) - torch.pi/2

        extrinsics = PoseSampler._get_extrinsics_from_batched_elev_azi(elevations, azimuths, radius)
        return extrinsics
    
    @staticmethod
    def sample_extrinsics_uniform(azimuth_mean, elevation_mean, azimuth_std=0, elevation_std=0, radius=1, batch_size=1, device='cpu'):
        """
        Samples points on a sphere from a uniform distribution and returns a camera pose.
        """
        azimuths = (torch.rand((batch_size, ), device=device) * 2 - 1 )* azimuth_std + azimuth_mean
        elevations = (torch.rand((batch_size, ), device=device) * 2 - 1 )* elevation_std + elevation_mean

        # correct the oversampling of the poles
        elevations = torch.clamp(elevations, -torch.pi/2 + 1e-5, torch.pi/2 - 1e-5)
        elevations = torch.arccos(- 2 * elevations/ torch.pi) - torch.pi/2

        extrinsics = PoseSampler._get_extrinsics_from_batched_elev_azi(elevations, azimuths, radius)
        return extrinsics  
    
    @staticmethod
    def sample_pose_gaussian(focal:float, azimuth_mean:float, elevation_mean:float, azimuth_std:float=0, elevation_std:float=0, radius:float=1, cx:float=0.5, cy:float=0.5, batch_size:int=1, device='cpu'):
        extrinsics = PoseSampler.sample_extrinsics_gaussian(azimuth_mean, elevation_mean, azimuth_std, elevation_std, radius, batch_size, device)
        intrinsics = PoseSampler.get_intrinsics(focal, cx, cy, batch_size, device)
        return torch.cat([extrinsics.reshape(batch_size, 16), intrinsics.reshape(batch_size, 9)], dim=1)

    @staticmethod
    def sample_pose_uniform(focal:float, azimuth_mean:float, elevation_mean:float, azimuth_std:float=0, elevation_std:float=0, radius:float=1, cx:float=0.5, cy:float=0.5, batch_size:int=1, device='cpu'):
        extrinsics = PoseSampler.sample_extrinsics_uniform(azimuth_mean, elevation_mean, azimuth_std, elevation_std, radius, batch_size, device)
        intrinsics = PoseSampler.get_intrinsics(focal, cx, cy, batch_size, device)
        return torch.cat([extrinsics.reshape(batch_size, 16), intrinsics.reshape(batch_size, 9)], dim=1)


class LazyPoseSampler(PoseSampler):
    "Lazy pose sampler that looks-up the meta of the given dataset to sample poses lazily. Anything lazy means that you do not not need to specify the intrinsics for every method"
    
    def __init__(self, dataset_name:str):
        metadata = DatasetIntrinsicsManager.get_intrinsics(dataset_name)
        self.focal = metadata.focal
        self.cx = metadata.cx
        self.cy = metadata.cy
        self.azimuth_range = metadata.azimuth_range
        self.elevation_range = metadata.elevation_range
        self.camera_distance = metadata.camera_distance

        self.azimuth_std = (metadata.azimuth_range[1] - metadata.azimuth_range[0])/2
        self.azimuth_mean = metadata.azimuth_range[0] + self.azimuth_std
        self.elevation_std = (metadata.elevation_range[1] - metadata.elevation_range[0])/2
        self.elevation_mean = metadata.elevation_range[0] + self.elevation_std

    def sample_pose(self, batch_size=1, device='cpu'): 
        """Sample a batch of poses using uniformly distributed extrinscs. The elevation and azimuth ranges, 
        as well as focal and camera distance are obtained from the dataset's metadata."""
        return PoseSampler.sample_pose_uniform(
            focal=self.focal,
            cx=self.cx,
            cy=self.cy,
            azimuth_mean=self.azimuth_mean,
            azimuth_std=self.azimuth_std,
            elevation_mean=self.elevation_mean,
            elevation_std=self.elevation_std,
            radius=self.camera_distance,
            batch_size=batch_size,
            device=device
        )

    def get_intrinsics(self, batch_size=1, device='cpu') :
        return PoseSampler.get_intrinsics(self.focal, self.cx, self.cy, batch_size, device)
    
    def get_pose(self, azimuth:Union[torch.Tensor, float], elevation:Union[torch.Tensor, float], radius:Union[torch.Tensor, float], batch_size=None, device='cpu') : 
        """
        Returns the 25-dimentional pose vector given azimuth, elevation and radius. Intrinsics are obtained from the dataset's metadata.
        Each of the extrinsics argument can be a scalar or a torch.Tensor of shape (batch_size, ).
        If only scalars are given, make sure to specify the batch_size. If one of the arguments is a torch.Tensor, the batch_size is inferred from it. 
        If several tensors are given, make sure they have the same batch_size.
        """
                
        return PoseSampler.get_pose(self.focal, azimuth, elevation, radius, self.cx, self.cy, batch_size, device)

    def sample_pose_gaussian(self, azimuth_mean:float, elevation_mean:float, azimuth_std:float=0, elevation_std:float=0, radius:float=1, batch_size:int=1, device='cpu'):
        return PoseSampler.sample_pose_gaussian(self.focal, azimuth_mean, elevation_mean, azimuth_std, elevation_std, radius, self.cx, self.cy, batch_size, device)
    
    def sample_pose_uniform(self, azimuth_mean:float, elevation_mean:float, azimuth_std:float=0, elevation_std:float=0, radius:float=1, batch_size:int=1, device='cpu'):
        return PoseSampler.sample_pose_uniform(self.focal, azimuth_mean, elevation_mean, azimuth_std, elevation_std, radius, self.cx, self.cy, batch_size, device)


