import copy
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)


class MultiHAtten(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(MultiHAtten, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        b, n, _, h = *q.shape, self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class CrossTransformer(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super(CrossTransformer, self).__init__()
        self.cross_attn = MultiHAtten(dim, heads=8, dim_head=64, dropout=dropout)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, source_x, target_x):
        target_x_tmp = self.cross_attn(target_x, source_x, source_x)
        target_x = target_x_tmp + target_x
        target_x = self.ffn(target_x) + target_x
        return target_x


class DyRout_block(nn.Module):
    def __init__(self, opt):
        super(DyRout_block, self).__init__()
        self.f_t = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size)
        self.f_v = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size)
        self.f_a = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size)

    def forward(self, source, t, v, a, senti):
        


class DyRoutTrans_block(nn.Module):
    def __init__(self, opt):
        super(DyRoutTrans_block, self).__init__()
        self.mhatt1 = DyRout_block(opt)
        self.mhatt2 = MultiHAtten(opt.hidden_size, dropout=0.3)
        self.ffn = FeedForward(opt.hidden_size, opt.ffn_size, dropout=0.)

        self.dropout1 = nn.Dropout(opt.dropout)
        self.norm1 = nn.LayerNorm(opt.hidden_size, eps=1e-6)

        self.dropout2 = nn.Dropout(opt.dropout)
        self.norm2 = nn.LayerNorm(opt.hidden_size, eps=1e-6)

        self.dropout3 = nn.Dropout(opt.dropout)
        self.norm3 = nn.LayerNorm(opt.hidden_size, eps=1e-6)

    def forward(self, i, t, v, a, mask, senti):
        x = self.norm1(x + self.dropout1(self.mhatt1(v=y, k=y, q=x, masks=y_masks)))
        x = self.norm2(x + self.dropout2(self.mhatt2(q=x, k=x, v=x)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x


class DyRoutTrans(nn.Module):
    def __init__(self, opt):
        super(DyRoutTrans, self).__init__()
        self.opt = opt

        # Length Align
        self.len_t = nn.Linear(opt.seq_lens[0], 39)
        self.len_v = nn.Linear(opt.seq_lens[1]+1, 39)
        self.len_a = nn.Linear(opt.seq_lens[2]+1, 39)

        # Dimension Align
        self.dim_t = nn.Linear(768*2, 256)
        self.dim_v = nn.Linear(256, 256)
        self.dim_a = nn.Linear(256, 256)

        self.pos_embedding_t = nn.parameter.Parameter(torch.randn(1, 39, 256))
        self.pos_embedding_v = nn.parameter.Parameter(torch.randn(1, 39, 256))
        self.pos_embedding_a = nn.parameter.Parameter(torch.randn(1, 39, 256))

        fusion_block = DyRoutTrans_block(opt)
        self.dec_list = self._get_clones(fusion_block, 3)

    def forward(self, uni_fea, uni_mask, uni_senti):
        hidden_t = self.len_t(self.dim_t(uni_fea['T']).permute(0, 2, 1)).permute(0, 2, 1)
        hidden_v = self.len_v(self.dim_v(uni_fea['V']).permute(0, 2, 1)).permute(0, 2, 1)
        hidden_a = self.len_a(self.dim_a(uni_fea['A']).permute(0, 2, 1)).permute(0, 2, 1)

        hidden_t = hidden_t + self.pos_embedding_t
        hidden_v = hidden_v + self.pos_embedding_v
        hidden_a = hidden_a + self.pos_embedding_a

        for i, dec in enumerate(self.dec_list):
            uni_fea = dec(i, source, hidden_t, hidden_v, hidden_a, uni_mask, uni_senti)
        return uni_fea

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SentiCLS(nn.Module):
    def __init__(self, opt):
        super(SentiCLS, self).__init__()
        self.fuse_layer = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.Sigmoid()
        )
        self.cls_layer = nn.Sequential(
            nn.Linear(256, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 1, bias=True)
        )

        self.layernorm = nn.LayerNorm(256)

    def forward(self, fusion_features):    # hidden_text, hidden_video, hidden_acoustic
        fusion_features = torch.mean(fusion_features, dim=-2)
        fusion_features = self.layernorm(fusion_features + self.fuse_layer(fusion_features))
        output = self.cls_layer(fusion_features)

        return output
