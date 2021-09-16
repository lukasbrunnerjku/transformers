import torch
from torch.utils import benchmark

from .modules import EncoderBlock

batch, seqlen, dmodel, h = 2, 9, 512, 8
hidden_mult, dropout = 4, 0.1
x = torch.randn(batch, seqlen, dmodel)
block = EncoderBlock(dmodel, h, hidden_mult, dropout)

t_cpu = benchmark.Timer(
    stmt='with torch.no_grad(): block(x)',
    globals={'x': x, 'block': block})

print(t_cpu.timeit(100))

block = EncoderBlock(dmodel, h, hidden_mult, dropout).cuda()
x = torch.randn(batch, seqlen, dmodel).cuda()

t_gpu = benchmark.Timer(
    stmt='with torch.no_grad(): block(x)',
    globals={'x': x, 'block': block})

print(t_gpu.timeit(100))


"""
<torch.utils.benchmark.utils.common.Measurement object at 0x0000029703D76160>
with torch.no_grad(): block(x)
  1.22 ms
  1 measurement, 100 runs , 1 thread
<torch.utils.benchmark.utils.common.Measurement object at 0x0000029703D76C10>
with torch.no_grad(): block(x)
  641.66 us
  1 measurement, 100 runs , 1 thread
"""
