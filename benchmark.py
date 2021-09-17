import torch
from torch.utils import benchmark

from .main_cnn import create_model as ResNet
from .main_transformer import create_model as ViT

for model in (ResNet(), ViT()):

    print(f"#### {model.__class__.__name__} ####")
    x = torch.randn(1, 3, 32, 32)  # CIFAR10

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params / 1e6 :.3f}M")

    t_cpu = benchmark.Timer(
        stmt="with torch.no_grad(): model(x)", globals={"x": x, "model": model}
    )

    print(t_cpu.timeit(100))

    x = torch.randn(1, 3, 32, 32).cuda()  # CIFAR10
    model.cuda()

    t_gpu = benchmark.Timer(
        stmt="with torch.no_grad(): model(x)", globals={"x": x, "model": model}
    )

    print(t_gpu.timeit(100))


"""
#### ResNet ####
Parameters: 11.174M
<torch.utils.benchmark.utils.common.Measurement object at 0x000001C70AA6D250>
with torch.no_grad(): model(x)
  15.06 ms
  1 measurement, 100 runs , 1 thread
<torch.utils.benchmark.utils.common.Measurement object at 0x000001C733442FD0>
with torch.no_grad(): model(x)
  2.86 ms
  1 measurement, 100 runs , 1 thread
#### VisionTransformer ####
Parameters: 26.355M
<torch.utils.benchmark.utils.common.Measurement object at 0x000001C73345C640>
with torch.no_grad(): model(x)
  11.85 ms
  1 measurement, 100 runs , 1 thread
<torch.utils.benchmark.utils.common.Measurement object at 0x000001C70AE59AF0>
with torch.no_grad(): model(x)
  5.58 ms
  1 measurement, 100 runs , 1 thread
"""
