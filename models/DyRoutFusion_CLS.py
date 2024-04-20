import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# --------------------------------
# ---- img Local Window Generator ----
# --------------------------------
def getImgMasks(scale=16, order=2):
    """
    :param scale: Feature Map Scale
    :param order: Local Window Size, e.g., order=2 equals to windows size (5, 5)
    :return: masks = (scale**2, scale**2)
    """
    masks = []
    _scale = scale
    assert order < _scale, 'order size be smaller than feature map scale'

    for i in range(_scale):
        for j in range(_scale):
            mask = np.ones([_scale, _scale], dtype=np.float32)
            for x in range(i - order, i + order + 1, 1):
                for y in range(j - order, j + order + 1, 1):
                    if (0 <= x < _scale) and (0 <= y < _scale):
                        mask[x][y] = 0
            mask = np.reshape(mask, [_scale * _scale])
            masks.append(mask)
    masks = np.array(masks)
    masks = np.asarray(masks, dtype=np.bool) # 0, 1 -> False True (True mask)
    return masks


def getMasks_img_multimodal(x_mask, __C):
    mask_list = [] # x_mask [64, 1, 1, 49]
    ORDERS = __C["ORDERS"]
    for order in ORDERS:
        if order == 0:
            mask_list.append(x_mask)
        else:
            mask_img = torch.from_numpy(getImgMasks(__C["IMG_SCALE"], order)).byte().to(x_mask.device) # (49, 49)
            mask = torch.cat([mask_img]*(__C["len"]//(__C["IMG_SCALE"]*__C["IMG_SCALE"])), dim=0) 
            mask = torch.cat([mask, mask_img[:(__C["len"]%(__C["IMG_SCALE"]*__C["IMG_SCALE"])),:]])
            mask = torch.logical_or(x_mask, mask) # (64, 1, max_len, grid_num)
            mask_list.append(mask)
    return mask_list


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


class FC(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0):
        super(FC, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0):
        super(MLP, self).__init__()
        self.fc = FC(input_dim, hidden_dim, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear(self.fc(x))


class cls_layer_img(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cls_layer_img, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, lang_feat, img_feat):
        proj_feat = self.proj_norm(img_feat)
        proj_feat = self.proj(proj_feat)
        return proj_feat


class cls_layer_txt(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cls_layer_txt, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, lang_feat, img_feat):
        proj_feat = self.proj_norm(lang_feat)
        proj_feat = self.proj(proj_feat)
        return proj_feat


class cls_layer_both(nn.Module):
    def __init__(self,  input_dim, output_dim):
        super(cls_layer_both, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, lang_feat, img_feat):
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)
        return proj_feat


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
            mask = self.make_mask(x).squeeze(1).squeeze(1).unsqueeze(2) # (8, 1, 1, 49) -> (8, 49, 1)
            scores = self.pool(x) # (8, 49, 1)
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
        alpha = (1 / self.len) * torch.ones(x.shape[0], self.len).to(x.device)# (bs, 4)
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

        alphas = self.routing_block(x, tau, masks) # (bs, 4)

        if self.opt["BINARIZE"]:
            if not training:
                alphas = self.argmax_binarize(alphas)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2) # (bs, 4, 49, 192)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2) # (bs, 4, 49, 192)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2) # (bs, 4, 49, 192)

        att_list = self.routing_att(v, k, q, masks) # (bs, order_num, head_num, grid_num, grid_num) (bs, 4, 4, 49, 49)
        att_map = torch.einsum('bl,blcnm->bcnm', alphas, att_list) # (bs, 4), (bs, 4, 4, 49, 49) - > (bs, 4, 49, 49)

        atted = torch.matmul(att_map, v) # (bs, 4, 49, [49]) * (bs, 4, [49],192) - > (bs, 4, 49, 192) mul [49, 49]*[49, 192], 

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.opt["hidden_size"]
        ) # (bs, 49, 768)

        atted = self.linear_merge(atted) # (bs, 4, 768)

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


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------
class FFN(nn.Module):
    def __init__(self, opt):
        super(FFN, self).__init__()

        self.mlp = MLP(
            input_dim=opt["hidden_size"],
            hidden_dim=opt["ffn_size"],
            output_dim=opt["hidden_size"],
            dropout=opt["dropout"],
            activation="ReLU"
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------
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

    def forward(self, x, y, y_masks, x_mask, tau, training): # x (64, 49, 512) y

        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=y, k=y, q=x, masks=y_masks, tau=tau, training=training)
        )) # (64, 49, 512) # (bs, 49, 768)

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


class DynRT_ED(nn.Module):
    def __init__(self, opt):
        super(DynRT_ED, self).__init__()
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
        x_masks = getMasks_img_multimodal(x_mask, self.opt)
        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for i, dec in enumerate(self.dec_list):
            y = dec(y, x, x_masks[:i+1], y_mask, self.tau, self.training) # (4, 360, 768)
        return y, x

    def set_tau(self, tau):
        self.tau = tau


class DynRT_E(nn.Module):
    def __init__(self, opt):
        super(DynRT_E, self).__init__()
        self.backbone = DynRT_ED(opt)
        if opt["classifier"] == 'both':
            self.cls_layer = cls_layer_both(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'text':
            self.cls_layer = cls_layer_txt(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'img':
            self.cls_layer = cls_layer_img(opt["hidden_size"],opt["output_size"])

    def forward(self, img_feat, lang_feat, lang_feat_mask):
        img_feat_mask = torch.zeros([img_feat.shape[0],1,1,img_feat.shape[1]],dtype=torch.bool,device=img_feat.device)
        # (bs, 1, 1, grid_num)
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = torch.mean(lang_feat, dim=1)
        img_feat = torch.mean(img_feat, dim=1)

        proj_feat = self.cls_layer(lang_feat, img_feat)

        return proj_feat, lang_feat, img_feat


class DynRT(torch.nn.Module):
    def __init__(self, bertl_text, vit, opt):
        super(DynRT, self).__init__()

        self.bertl_text = bertl_text
        self.opt = opt
        self.vit=vit
        self.input1=opt["input1"]
        self.input2=opt["input2"]
        self.input3=opt["input3"]

        self.trar = model.TRAR.DynRT_E(opt)
        self.sigm = torch.nn.Sigmoid()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(opt["output_size"],2)
        )

    def vit_forward(self,x):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x[:,1:]

    # forward propagate input
    def forward(self, input):
        # (bs, max_len, dim)
        bert_embed_text = self.bertl_text.embeddings(input_ids = input[self.input1])
        # (bs, max_len, dim)
        # bert_text = self.bertl_text.encoder.layer[0](bert_embed_text)[0]
        for i in range(self.opt["roberta_layer"]):
            bert_text = self.bertl_text.encoder.layer[i](bert_embed_text)[0]
            bert_embed_text = bert_text
        # (bs, grid_num, dim)
        img_feat = self.vit_forward(input[self.input2])

        (out1, lang_emb, img_emb) = self.trar(img_feat, bert_embed_text,input[self.input3].unsqueeze(1).unsqueeze(2))

        out = self.classifier(out1)
        result = self.sigm(out)

        del bert_embed_text, bert_text, img_feat, out1, out

        return result, lang_emb, img_emb


class DyRoutTrans(nn.Module):
    def __init__(self, opt):
        super(DyRoutTrans, self).__init__()
        # Multimodal Fusion and Classfication Module
        # self.fusion_layer = CrossTransformer(source_num_frames=8, tgt_num_frames=8, dim=128, depth=fusion_layer_depth, heads=8, mlp_dim=128)

    def forward(self, unimodal_features, unimodal_senti):
        return unimodal_features


class SentiCLS(nn.Module):
    def __init__(self, opt):
        super(SentiCLS, self).__init__()
        # Multimodal Fusion and Classfication Module
        # self.fusion_layer = CrossTransformer(source_num_frames=8, tgt_num_frames=8, dim=128, depth=fusion_layer_depth, heads=8, mlp_dim=128)
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
