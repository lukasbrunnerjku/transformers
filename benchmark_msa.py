import torch
from torch.utils import benchmark

from .modules import MSA

batch, seqlen, dmodel, h = 2, 9, 512, 8

x = torch.randn(batch, seqlen, dmodel)
msa = MSA(dmodel, h)

t_cpu = benchmark.Timer(
    stmt='with torch.no_grad(): msa.forward(x)',
    globals={'x': x, 'msa': msa})

print(t_cpu.timeit(100))

msa = MSA(dmodel, h).cuda()
x = torch.randn(batch, seqlen, dmodel).cuda()

t_gpu = benchmark.Timer(
    stmt='with torch.no_grad(): msa.forward(x)',
    globals={'x': x, 'msa': msa})

print(t_gpu.timeit(100))


"""
<torch.utils.benchmark.utils.common.Measurement object at 0x00000259E727CB50>
with torch.no_grad(): msa.forward(x)
  558.87 us
  1 measurement, 100 runs , 1 thread
<torch.utils.benchmark.utils.common.Measurement object at 0x00000259E72F7070>
with torch.no_grad(): msa.forward(x)
  282.22 us
  1 measurement, 100 runs , 1 thread
"""
