import torch
from .modules import *


def test_EncoderBlock_forward():

    batch, seqlen, dmodel, h = 2, 9, 512, 8
    hidden_mult, dropout = 4, 0.1
    block = EncoderBlock(dmodel, h, hidden_mult, dropout)
    x = torch.randn(batch, seqlen, dmodel)
    y = block(x)
    assert y.shape == x.shape


def test_MSA_forward():

    batch, seqlen, dmodel, h = 2, 9, 512, 8
    msa = MSA(dmodel, h)
    x = torch.randn(batch, seqlen, dmodel)
    y = msa(x)
    assert y.shape == x.shape


def compare_MSA_forward():

    batch, seqlen, dmodel, h = 2, 9, 512, 8
    msa = MSA(dmodel, h)
    x = torch.randn(batch, seqlen, dmodel)

    y = msa.forward(x)
    y_einsum = msa.forward_einsum(x)

    print(torch.allclose(y, y_einsum))


def test_VisionTransformer_forward():

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
    seqlen = int(H / patch_size * W / patch_size)
    print(f"Sequence length: {seqlen}")

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
        n_classes,
    )

    print(
        f"Parameters: {sum(p.numel() for p in vit.parameters() if p.requires_grad) / 1e6 :.3f}M"
    )

    logits = vit(x)
    assert logits.shape == (batch, n_classes)


if __name__ == "__main__":

    test_MSA_forward()
    compare_MSA_forward()
    test_EncoderBlock_forward()
    test_VisionTransformer_forward()
