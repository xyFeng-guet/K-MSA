import math
import copy
import torch
from torch import nn
import torch.nn.functional as F


# 重点更改动态路由融合
class SARoutingBlock(nn.Module):
    """Self-Attention Routing Block
    """
    def __init__(self, opt):
        super(SARoutingBlock, self).__init__()
        self.opt = opt
        self.linear_v = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_k = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_q = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_merge = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, q, k, v, masks):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt.hidden_size / self.opt["multihead"])
        ).transpose(1, 2)       # (bs, 4, 49, 192)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt.hidden_size / self.opt["multihead"])
        ).transpose(1, 2)       # (bs, 4, 49, 192)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt.hidden_size / self.opt["multihead"])
        ).transpose(1, 2)       # (bs, 4, 49, 192)

        att_list = self.routing_att(v, k, q, masks)     # (bs, order_num, head_num, grid_num, grid_num) (bs, 4, 4, 49, 49)
        att_map = torch.einsum('bl,blcnm->bcnm', alphas, att_list)      # (bs, 4), (bs, 4, 4, 49, 49) - > (bs, 4, 49, 49)

        atted = torch.matmul(att_map, v)        # (bs, 4, 49, [49]) * (bs, 4, [49],192) - > (bs, 4, 49, 192) mul [49, 49]*[49, 192]

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.opt.hidden_size
        )       # (bs, 49, 768)

        atted = self.linear_merge(atted)        # (bs, 4, 768)

        return atted

    def routing_att(self, value, key, query, masks):
        d_k = query.size(-1) # masks [[bs, 1, 1, 49], [bs, 1, 49, 49], [bs, 1, 49, 49], [bs, 1, 49, 49]]
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k) # (bs, 4, 49, 49) (2, 4, 360, 49)
        # k q v [4, 4, 49, 192] key (2, 4, 49, 192) query [2, 4, 360, 192]
        for i in range(len(masks)):
            mask = masks[i] # (bs, 1, 49, 49)
            scores_temp = scores.masked_fill(mask, -1e9)
            att_map = F.softmax(scores_temp, dim=-1)
            att_map = self.dropout(att_map)
            if i == 0:
                att_list = att_map.unsqueeze(1) # (bs, 1, 4, 49, 49)
            else:
                att_list = torch.cat((att_list, att_map.unsqueeze(1)), 1)  # (bs, 2, 4, 49, 49) -> (bs, 3, 4, 49, 49)

        return att_list


class FFN(nn.Module):
    def __init__(self, opt):
        super(FFN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.ffn_size),
            nn.ReLU(inplace=True),
            nn.Dropout(opt.dropout)
        )
        self.linear = nn.Linear(opt.ffn_size, opt.hidden_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class MHAtt(nn.Module):
    def __init__(self, opt):
        super(MHAtt, self).__init__()
        self.opt = opt

        self.linear_v = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_k = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_q = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_merge = nn.Linear(opt.hidden_size, opt.hidden_size)

        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt.hidden_size / self.opt["multihead"])
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt.hidden_size / self.opt["multihead"])
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt.hidden_size / self.opt["multihead"])
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.opt.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value)


class multiTRAR_SA_block(nn.Module):
    def __init__(self, opt):
        super(multiTRAR_SA_block, self).__init__()
        # self.mhatt1 = SARoutingBlock(opt)
        self.mhatt2 = MHAtt(opt)
        self.ffn = FFN(opt)

        self.dropout1 = nn.Dropout(opt.dropout)
        self.norm1 = nn.LayerNorm(opt.hidden_size, eps=1e-6)

        self.dropout2 = nn.Dropout(opt.dropout)
        self.norm2 = nn.LayerNorm(opt.hidden_size, eps=1e-6)

        self.dropout3 = nn.Dropout(opt.dropout)
        self.norm3 = nn.LayerNorm(opt.hidden_size, eps=1e-6)

    def forward(self, x, y, x_mask, y_masks):
        x = self.norm1(x + self.dropout1(self.mhatt1(v=y, k=y, q=x, masks=y_masks)))
        x = self.norm2(x + self.dropout2(self.mhatt2(v=x, k=x, q=x, mask=x_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x


class DyRoutTrans(nn.Module):
    def __init__(self, opt):
        super(DyRoutTrans, self).__init__()
        self.opt = opt
        fusion_block = multiTRAR_SA_block(opt)
        self.dec_list = self._get_clones(fusion_block, 4)

        # Length Align
        self.len_t = nn.Linear(opt.seq_len, 39)
        self.len_v = nn.Linear(opt.seq_len, 39)
        self.len_a = nn.Linear(opt.seq_len, 39)

    def forward(self, uni_fea, uni_mask, uni_senti):
        # y text (bs, max_len, dim) x img (bs, gird_num, dim) y_mask (bs, 1, 1, max_len) x_mask (bs, 1, 1, grid_num)
        for i, dec in enumerate(self.dec_list):
            fuse_fea = dec(uni_fea, fuse_fea, uni_mask, uni_senti)   # (4, 360, 768)
        return y, x

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SentiCLS(nn.Module):
    def __init__(self, opt):
        super(SentiCLS, self).__init__()
        self.fuse_layer = nn.Sequential(
            nn.Linear((768 + 128 * 2) * 2, (768 + 128 * 2) * 2, bias=True),
            nn.ReLU(),
            nn.Linear((768 + 128 * 2) * 2, (768 + 128 * 2) * 2, bias=True),
            nn.Sigmoid()
        )
        self.cls_layer = nn.Sequential(
            nn.Linear((768 + 128 * 2) * 2, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 1, bias=True)
        )

        self.layernorm = nn.LayerNorm((768 + 128 * 2) * 2)

    def forward(self, hidden_text, hidden_video, hidden_acoustic):
        h_t_global = hidden_text[:, 0, :]   # torch.mean(hidden_text, dim=1)
        h_v_global = torch.mean(hidden_video, dim=1)
        h_a_global = torch.mean(hidden_acoustic, dim=1)

        fusion_features = torch.cat((h_t_global, h_v_global, h_a_global), dim=-1)
        fusion_features = self.layernorm(fusion_features + self.fuse_layer(fusion_features))
        output = self.cls_layer(fusion_features)

        return output
