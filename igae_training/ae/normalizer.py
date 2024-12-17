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

def renormalize_img(x):
    "expect image in [-1,1] range and returns it in [0,1] range"
    return 0.5 * (x + 1)

def denormalize_img(x):
    "expect image in [0,1] range and returns it in [-1,1] range"
    return 2 * x - 1

class TanhNormalizer():

    def __init__(self, scale=0.02, eps=1e-6):
        self.scale = scale
        self.eps = eps
        self.tanh = torch.nn.Tanh()
    
    def normalize(self, x):
        "Expect input in the latent space range and maps it in [-1,1]"
        return self.tanh(self.scale * x)

    def denormalize(self, x):
        "Expect input in the [-1, 1] range and maps in in the latent space range"
        x = x.clamp(-1 + self.eps, 1 - self.eps)
        return (1/self.scale) * torch.atanh(x)