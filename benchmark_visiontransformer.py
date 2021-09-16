import torch
from torch.utils import benchmark

from .modules import VisionTransformer

dmodel = 512
h = 8
hidden_mult = 4
dropout = 0.1
patch_size = 16
in_channels = 3
L = 5
H, W = 224, 224
batch = 2
n_hidden = 1024
n_classes = 100

assert H % patch_size == 0
assert W % patch_size == 0
seqlen = int(H/patch_size * W/patch_size)
print(f'Sequence length: {seqlen}')

x = torch.randn(batch, in_channels, H, W)

vit = VisionTransformer(
    dmodel,
    
    # EncoderBlocks
    h,
    hidden_mult, 
    dropout, 
    L,

    # PatchEmbeddings
    patch_size,
    in_channels,
    
    # ClassificationHead
    n_hidden, 
    n_classes
)
print(f'Parameters: {sum(p.numel() for p in vit.parameters() if p.requires_grad) / 1e6 :.3f}M')

t_cpu = benchmark.Timer(
    stmt='with torch.no_grad(): vit(x)',
    globals={'x': x, 'vit': vit})

print(t_cpu.timeit(100))

x = torch.randn(batch, in_channels, H, W).cuda()

vit = VisionTransformer(
    dmodel,
    
    # EncoderBlocks
    h,
    hidden_mult, 
    dropout, 
    L,

    # PatchEmbeddings
    patch_size,
    in_channels,
    
    # ClassificationHead
    n_hidden, 
    n_classes
).cuda()

t_gpu = benchmark.Timer(
    stmt='with torch.no_grad(): vit(x)',
    globals={'x': x, 'vit': vit})

print(t_gpu.timeit(100))


"""
Sequence length: 196
Parameters: 17.289M
<torch.utils.benchmark.utils.common.Measurement object at 0x0000024AA25F0D60>
with torch.no_grad(): vit(x)
  127.13 ms
  1 measurement, 100 runs , 1 thread
<torch.utils.benchmark.utils.common.Measurement object at 0x0000024AA2564340>
with torch.no_grad(): vit(x)
  3.01 ms
  1 measurement, 100 runs , 1 thread
"""
