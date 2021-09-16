import torch
from torch import nn
import torch.nn.functional as F
import math


class MSA(nn.Module):
    def __init__(self, dmodel, h):
        super().__init__()

        self.dmodel = dmodel
        self.h = h
        self.toK = nn.Linear(dmodel, dmodel, bias=False)
        self.toQ = nn.Linear(dmodel, dmodel, bias=False)
        self.toV = nn.Linear(dmodel, dmodel, bias=False)
        self.unify = nn.Linear(dmodel, dmodel)

    def forward(self, x):

        b, n, dmodel = x.size()
        h = self.h
        d = int(dmodel / h)

        K = self.toK(x).view(b, n, h, d)
        Q = self.toQ(x).view(b, n, h, d)
        V = self.toV(x).view(b, n, h, d)

        K = K.transpose(1, 2).contiguous().view(b * h, n, d)
        Q = Q.transpose(1, 2).contiguous().view(b * h, n, d)
        V = V.transpose(1, 2).contiguous().view(b * h, n, d)

        dot = torch.bmm(Q, K.transpose(1, 2))  # b*h x n x n = b*h x n x d @ b*h x d x n
        dot = dot / math.sqrt(d)  # b*h x n x n
        dot = F.softmax(dot, dim=2)  # b*h x n x n

        out = torch.bmm(dot, V).view(
            b, h, n, d
        )  # b*h x n x d = b*h x n x n @ b*h x n x d
        out = out.transpose(1, 2).contiguous().view(b, n, h * d)

        return self.unify(out)  # b x n x dmodel

    def forward_einsum(self, x):

        b, n, dmodel = x.size()
        h = self.h
        d = int(dmodel / h)

        K = self.toK(x).view(b, n, h, d)
        Q = self.toQ(x).view(b, n, h, d)
        V = self.toV(x).view(b, n, h, d)

        dot = torch.einsum("bthe,bihe->bhti", Q, K) / math.sqrt(d)  # b x h x n x n
        dot = F.softmax(dot, dim=-1)  # b x h x n x n

        out = torch.einsum("bhtd,bdhe->bthe", dot, V)  # b x n x h x d
        out = out.reshape(b, n, -1)  # b x n x h*d
        return self.unify(out)  # b x n x dmodel


class EncoderBlock(nn.Module):
    def __init__(self, dmodel: int, h: int, hidden_mult, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(dmodel)
        self.msa = MSA(dmodel, h)
        self.do1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(dmodel)
        self.mlp = nn.Sequential(
            nn.Linear(dmodel, hidden_mult * dmodel),
            nn.GELU(),
            nn.Linear(hidden_mult * dmodel, dmodel),
        )
        self.do2 = nn.Dropout(dropout)

    def forward(self, x):

        z = self.do1(self.msa(self.ln1(x))) + x
        return self.do2(self.msa(self.ln1(z))) + z


class PatchEmbeddings(nn.Module):
    def __init__(self, dmodel: int, patch_size: int, in_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, dmodel, patch_size, stride=patch_size)

    def forward(self, x):

        x = self.conv(x)
        b, dmodel, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # b x h x w x dmodel
        return x.view(b, h * w, dmodel)  # b x h*w x dmodel


class PositionalEmbeddings(nn.Module):
    def __init__(self, dmodel: int, max_len: int = 1000):
        super().__init__()

        self.pos_enc = nn.Parameter(torch.zeros(1, max_len, dmodel), requires_grad=True)

    def forward(self, x):  # b x h*w x dmodel

        seqlen = x.shape[1]
        return x + self.pos_enc[:, :seqlen, :]  # b x h*w x dmodel


class ClassificationHead(nn.Module):
    def __init__(self, dmodel: int, n_hidden: int, n_classes: int):
        super().__init__()

        self.linear1 = nn.Linear(dmodel, n_hidden)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x):  # b x dmodel

        return self.linear2(self.act(self.linear1(x)))  # b x n_classes (logits)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        dmodel: int,
        # EncoderBlocks
        h: int,
        hidden_mult: int,
        dropout: float,
        L: int,
        # PatchEmbeddings
        patch_size: int,
        in_channels: int,
        # ClassificationHead
        n_hidden: int,
        n_classes: int,
    ):
        super().__init__()

        self.patch_emb = PatchEmbeddings(dmodel, patch_size, in_channels)
        self.pos_emb = PositionalEmbeddings(dmodel)
        self.classification = ClassificationHead(dmodel, n_hidden, n_classes)
        self.blocks = nn.ModuleList(
            [EncoderBlock(dmodel, h, hidden_mult, dropout) for _ in range(L)]
        )
        self.cls_emb = nn.Parameter(torch.randn(1, 1, dmodel), requires_grad=True)
        self.ln = nn.LayerNorm(dmodel)

    def forward(self, x):  # b x c x H x W

        x = self.patch_emb(x)  # b x h*w x dmodel
        x = self.pos_emb(x)  # b x h*w x dmodel

        b = x.shape[0]
        cls_emb = self.cls_emb.expand(b, -1, -1)  # b x 1 x dmodel
        z = torch.cat([cls_emb, x], dim=1)  # b x (h*w)+1 x dmodel

        for block in self.blocks:
            z = block(z)

        y = self.ln(z[:, 0, :])  # b x dmodel
        return self.classification(y)  # b x n_classes (logits)
