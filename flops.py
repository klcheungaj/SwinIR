

from models.network_swinir import SwinIR as net
import torch
from torch import nn
from thop import profile

model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')


shape = (3, 540, 960)
input = torch.randn((1, shape[0], shape[1], shape[2]))
flop, param = profile(model, (input,))
print(flop/10**9, param)