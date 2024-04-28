import math
import copy
import torch
from torch import nn
import torch.nn.functional as F


class SoftRoutingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pooling='attention', reduction=2):
        super(SoftRoutingBlock, self).__init__()
        self.pooling = pooling

        if pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == 'fc':
            self.pool = nn.Linear(in_channel, 1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=True),
        )

    def forward(self, x, tau, masks):
        if self.pooling == 'avg':
            x = x.transpose(1, 2)
            x = self.pool(x)
            logits = self.mlp(x.squeeze(-1))
        elif self.pooling == 'fc':
            b, _, c = x.size()
            mask = self.make_mask(x).squeeze(1).squeeze(1).unsqueeze(2)     # (8, 1, 1, 49) -> (8, 49, 1)
            scores = self.pool(x)       # (8, 49, 1)
            scores = scores.masked_fill(mask, -1e9)
            scores = F.softmax(scores, dim=1)
            _x = x.mul(scores)
            x = torch.sum(_x, dim=1)
            logits = self.mlp(x)

        alpha = F.softmax(logits, dim=-1)  #
        return alpha

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


class HardRoutingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pooling='attention', reduction=2):
        super(HardRoutingBlock, self).__init__()
        self.pooling = pooling

        if pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == 'fc':
            self.pool = nn.Linear(in_channel, 1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=True),
        )

    def forward(self, x, tau, masks):
        if self.pooling == 'avg':
            x = x.transpose(1, 2)
            x = self.pool(x)
            logits = self.mlp(x.squeeze(-1))
        elif self.pooling == 'fc':
            b, _, c = x.size()
            mask = self.make_mask(x).squeeze(1).squeeze(1).unsqueeze(2)
            scores = self.pool(x)
            scores = scores.masked_fill(mask, -1e9)
            scores = F.softmax(scores, dim=1)
            _x = x.mul(scores)
            x = torch.sum(_x, dim=1)
            logits = self.mlp(x)

        alpha = self.gumbel_softmax(logits, -1, tau)
        return alpha

    def gumbel_softmax(self, logits, dim=-1, temperature=0.1):
        '''
        Use this to replace argmax
        My input is probability distribution, multiply by 10 to get a value like logits' outputs.
        '''
        gumbels = -torch.empty_like(logits).exponential_().log()
        logits = (logits.log_softmax(dim=dim) + gumbels) / temperature
        return F.softmax(logits, dim=dim)

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


class mean_Block(nn.Module):
    """
    Self-Attention Routing Block
    """
    def __init__(self, hidden_size, orders):
        super(mean_Block, self).__init__()
        self.len = orders
        self.hidden_size = hidden_size

    def forward(self, x, tau, masks):
        alpha = (1 / self.len) * torch.ones(x.shape[0], self.len).to(x.device)      # (bs, 4)
        return alpha


class SARoutingBlock(nn.Module):
    """
    Self-Attention Routing Block
    """
    def __init__(self, opt):
        super(SARoutingBlock, self).__init__()
        self.opt = opt

        self.linear_v = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        self.linear_k = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        self.linear_q = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        self.linear_merge = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        if opt["routing"] == 'hard':
            self.routing_block = HardRoutingBlock(opt["hidden_size"], opt["orders"], opt["pooling"])
        elif opt["routing"] == 'soft':
            self.routing_block = SoftRoutingBlock(opt["hidden_size"], opt["orders"], opt["pooling"])
        elif opt["routing"] == 'mean':
            self.routing_block = mean_Block(opt["hidden_size"], opt["orders"])

        self.dropout = nn.Dropout(opt["dropout"])

    def forward(self, v, k, q, masks, tau, training):
        n_batches = q.size(0)
        x = v

        alphas = self.routing_block(x, tau, masks)      # (bs, 4)

        if self.opt["BINARIZE"]:
            if not training:
                alphas = self.argmax_binarize(alphas)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2)       # (bs, 4, 49, 192)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2)       # (bs, 4, 49, 192)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2)       # (bs, 4, 49, 192)

        att_list = self.routing_att(v, k, q, masks)     # (bs, order_num, head_num, grid_num, grid_num) (bs, 4, 4, 49, 49)
        att_map = torch.einsum('bl,blcnm->bcnm', alphas, att_list)      # (bs, 4), (bs, 4, 4, 49, 49) - > (bs, 4, 49, 49)

        atted = torch.matmul(att_map, v)        # (bs, 4, 49, [49]) * (bs, 4, [49],192) - > (bs, 4, 49, 192) mul [49, 49]*[49, 192]

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.opt["hidden_size"]
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

    def argmax_binarize(self, alphas):
        n = alphas.size()[0]
        out = torch.zeros_like(alphas)
        indexes = alphas.argmax(-1)
        out[torch.arange(n), indexes] = 1
        return out


class MHAtt(nn.Module):
    def __init__(self, opt):
        super(MHAtt, self).__init__()
        self.opt = opt

        self.linear_v = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        self.linear_k = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        self.linear_q = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        self.linear_merge = nn.Linear(opt["hidden_size"], opt["hidden_size"])

        self.dropout = nn.Dropout(opt["dropout"])

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.opt["hidden_size"]
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


class FFN(nn.Module):
    def __init__(self, opt):
        super(FFN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(opt["hidden_size"], opt["hidden_size"]),
            nn.ReLU(inplace=True),
            nn.Dropout(opt["dropout"])
        )
        self.linear = nn.Linear(opt["ffn_size"], opt["hidden_size"])

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones(size))
        self.b = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class multiTRAR_SA_block(nn.Module):
    def __init__(self, opt):
        super(multiTRAR_SA_block, self).__init__()
        self.mhatt1 = SARoutingBlock(opt)
        self.mhatt2 = MHAtt(opt)
        self.ffn = FFN(opt)

        self.dropout1 = nn.Dropout(opt["dropout"])
        self.norm1 = LayerNorm(opt["hidden_size"])

        self.dropout2 = nn.Dropout(opt["dropout"])
        self.norm2 = LayerNorm(opt["hidden_size"])

        self.dropout3 = nn.Dropout(opt["dropout"])
        self.norm3 = LayerNorm(opt["hidden_size"])

    def forward(self, x, y, x_mask, y_masks, tau, training):
        x = self.norm1(x + self.dropout1(self.mhatt1(v=y, k=y, q=x, masks=y_masks, tau=tau, training=training)))
        x = self.norm2(x + self.dropout2(self.mhatt2(v=x, k=x, q=x, mask=x_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x


class DynRT_E(nn.Module):
    def __init__(self, opt):
        super(DynRT_E, self).__init__()
        self.opt = opt
        self.tau = opt["tau_max"]
        opt_list = []
        for i in range(opt["layer"]):
            opt_copy = copy.deepcopy(opt)
            opt_copy["ORDERS"] = opt["ORDERS"][:len(opt["ORDERS"])-i]
            opt_copy["orders"] = len(opt["ORDERS"])-i
            opt_list.append(copy.deepcopy(opt_copy))
        self.dec_list = nn.ModuleList([multiTRAR_SA_block(opt_list[-(i+1)]) for i in range(opt["layer"])])

    def forward(self, y, x, y_mask, x_mask):
        # y text (bs, max_len, dim) x img (bs, gird_num, dim) y_mask (bs, 1, 1, max_len) x_mask (bs, 1, 1, grid_num)
        # Input encoder last hidden vector and obtain decoder last hidden vectors
        for i, dec in enumerate(self.dec_list):
            y = dec(x, y, x_mask, y_mask, self.tau, self.training)   # (4, 360, 768)
        return y, x


class DyRoutTrans(nn.Module):
    def __init__(self, opt):
        super(DyRoutTrans, self).__init__()
        self.opt = opt
        self.multifuse = DynRT_E(opt)
        self.cls_layer = nn.Sequential(
            LayerNorm(opt["hidden_size"]),
            nn.Linear(opt["hidden_size"], opt["output_size"])
        )
        # self.fusion_layer = CrossTransformer(source_num_frames=8, tgt_num_frames=8, dim=128, depth=fusion_layer_depth, heads=8, mlp_dim=128)

    def forward(self, lang_feat, img_feat, inputs, unimodal_senti):
        lang_feat_mask = inputs[self.input3].unsqueeze(1).unsqueeze(2)
        img_feat_mask = torch.zeros([img_feat.shape[0], 1, 1, img_feat.shape[1]], dtype=torch.bool, device=img_feat.device)     # (bs, 1, 1, grid_num)
        lang_feat, img_feat = self.multifuse(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_emb = torch.mean(lang_feat, dim=1)
        img_emb = torch.mean(img_feat, dim=1)
        result = self.cls_layer(lang_emb + img_emb)

        return lang_emb, img_emb, result


class SentiCLS(nn.Module):
    def __init__(self, opt):
        super(SentiCLS, self).__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 1, bias=True)
        )

    def forward(self, hidden_text, hidden_video, hidden_acoustic):
        h_v_global = torch.mean(hidden_video, dim=1)
        h_a_global = torch.mean(hidden_acoustic, dim=1)
        h_t_global = torch.mean(hidden_text, dim=1)

        fusion_features = torch.cat((h_v_global, h_a_global, h_t_global), dim=-1)
        output = self.fusion_layer(fusion_features)

        return output
