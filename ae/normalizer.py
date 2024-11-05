# Sliding window statistics
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