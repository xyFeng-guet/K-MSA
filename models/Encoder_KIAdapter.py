import torch
from torch import nn, einsum
from einops import rearrange
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from transformers import BertConfig, BertModel, BertTokenizer


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out):
        super(Classifier, self).__init__()
        ModuleList = []
        for i, h in enumerate(hidden_size):
            if i == 0:
                ModuleList.append(nn.Linear(input_size, h))
                ModuleList.append(nn.GELU())
            else:
                ModuleList.append(nn.Linear(hidden_size[i - 1], h))
                ModuleList.append(nn.GELU())
        ModuleList.append(nn.Linear(hidden_size[-1], output_size))

        self.MLP = nn.Sequential(*ModuleList)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(x)
        output = self.MLP(x)
        return output


class FeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, drop_out=0.1):
        super(FeatureProjector, self).__init__()
        self.feed_foward_size = int(output_dim / 2)
        self.project_size = output_dim - self.feed_foward_size
        self.proj1 = nn.Linear(input_dim, self.feed_foward_size, bias=True)

        self.proj2 = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.proj2.append(nn.Linear(input_dim, self.project_size, bias=False))
            else:
                self.proj2.append(nn.Linear(self.project_size, self.project_size, bias=False))
            self.proj2.append(nn.GELU())

        self.layernorm_ff = nn.LayerNorm(self.feed_foward_size)
        self.layernorm = nn.LayerNorm(self.project_size)
        self.MLP = nn.Sequential(*self.proj2)
        self.drop = nn.Dropout(p=drop_out)

    def forward(self, batch):
        # input: list of data samples with different seq length
        dropped = self.drop(batch)
        ff = self.proj1(dropped)
        x = self.MLP(dropped)
        x = torch.cat([self.layernorm(x), self.layernorm_ff(ff)], dim=-1)
        # return x.transpose(0, 1)  # return shape: [seq,batch,fea]
        return x


class PositionEncoding(nn.Module):
    """Construct the CLS token, position and patch embeddings.
    """
    def __init__(self, num_patches, fea_size, tf_hidden_dim, drop_out):
        super(PositionEncoding, self).__init__()
        self.cls_token = nn.Parameter(torch.ones(1, 1, tf_hidden_dim))
        self.proj = nn.Linear(fea_size, tf_hidden_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, tf_hidden_dim))
        self.dropout = nn.Dropout(drop_out)

    def forward(self, embeddings):
        batch_size = embeddings.shape[0]
        embeddings = self.proj(embeddings)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class MixDomainAdapter(nn.Module):
    def __init__(self, up_prj, down_prj, nhead=4, num_layers=2, dropout=0.1):
        super(MixDomainAdapter, self).__init__()
        self.down_project = nn.Linear(up_prj, down_prj)
        self.up_project = nn.Linear(down_prj, up_prj)

        tfencoderlayer = TransformerEncoderLayer(down_prj, nhead, down_prj // 2, dropout=dropout, activation='gelu', batch_first=True)
        self.tfencoder = TransformerEncoder(tfencoderlayer, num_layers)

        self.laynorm = nn.LayerNorm(up_prj)

    def forward(self, ex_know, src_key_padding_mask):
        hidden = self.down_project(ex_know)
        hidden = self.tfencoder(hidden, mask=None, src_key_padding_mask=src_key_padding_mask)[0]
        domain_know = self.up_project(hidden)
        output = domain_know + ex_know
        return output


class TfEncoder(nn.Module):
    def __init__(self, fea_size, num_patches, nhead, dim_feedforward, num_layers, pos_dropout=0., tf_dropout=0.2):
        super(TfEncoder, self).__init__()
        self.pos_encoder = PositionEncoding(
            num_patches=num_patches,
            fea_size=fea_size,
            tf_hidden_dim=dim_feedforward,
            drop_out=pos_dropout
        )

        tfencoderlayer = TransformerEncoderLayer(dim_feedforward, nhead, dim_feedforward // 2, dropout=tf_dropout, activation='gelu', batch_first=True)
        self.tfencoder = TransformerEncoder(tfencoderlayer, num_layers)

    def forward(self, src, src_key_padding_mask):
        src = self.pos_encoder(src)
        output, hidden_list = self.tfencoder(src, mask=None, src_key_padding_mask=src_key_padding_mask)
        return output


class UniEncoder(nn.Module):
    def __init__(self, m, pretrained, num_patches, fea_size, nhead, dim_feedforward, num_layers):
        super(UniEncoder, self).__init__()
        self.m = m

        if m in "VA":
            self.tfencoder = TfEncoder(
                fea_size=fea_size,
                num_patches=num_patches,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_layers=num_layers
            )
            self.layernorm = nn.LayerNorm(dim_feedforward)
        else:
            self.model_config = BertConfig.from_pretrained(pretrained, output_hidden_states=True)
            self.tokenizer = BertTokenizer.from_pretrained(pretrained, do_lower_case=True)
            self.model = BertModel.from_pretrained(pretrained, config=self.model_config)
            # self.projector = FeatureProjector(fea_size, dim_feedforward)

        self.adapter = MixDomainAdapter(up_prj=dim_feedforward, down_prj=dim_feedforward // 2)

    def forward(self, inputs, key_padding_mask):
        if self.m in "VA":
            x = self.tfencoder(inputs, src_key_padding_mask=key_padding_mask)
            hidden_state = self.layernorm(x)
            mean_state = torch.mean(x, dim=-2)
            return hidden_state, mean_state
        else:
            input_ids, input_mask, segment_ids = inputs[:, 0, :].long(), inputs[:, 1, :].float(), inputs[:, 2, :].long()    # 更换原始文本，使用tokenizer
            last_hidden_states = self.model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids
            )  # type: ignore # Models outputs are now tuples
            hidden_state = last_hidden_states[0]
            cls_token = hidden_state[:, 0, :]
            # hidden_text = self.projector(hidden_text)
            return hidden_state, cls_token

    def get_tokenizer(self):
        return self.tokenizer


class UniPretrain(nn.Module):
    def __init__(self, modality, num_patches, pretrained='./BERT/bert-base-chinese', fea_size=709, proj_fea_dim=128, drop_out=0.1):
        super(UniPretrain, self).__init__()
        self.m = modality
        if modality == "T":
            proj_fea_dim = 768

        self.encoder = UniEncoder(
            m=modality,
            pretrained=pretrained,
            fea_size=fea_size,
            num_patches=num_patches,
            nhead=8,
            dim_feedforward=proj_fea_dim,
            num_layers=2
        )
        self.decoder = Classifier(
            input_size=proj_fea_dim,
            hidden_size=[int(proj_fea_dim / 2), int(proj_fea_dim / 4), int(proj_fea_dim / 8)],
            output_size=1,
            drop_out=drop_out
        )

    def forward(self, uni_fea):
        uni_fea, key_padding_mask = uni_fea[self.m], uni_fea["mask"][self.m]
        hidden_state, sentence_state = self.encoder(uni_fea, key_padding_mask)
        pred = self.decoder(sentence_state)
        return hidden_state, pred


class UnimodalEncoder(nn.Module):
    def __init__(self, opt, bert_pretrained='./BERT/bert-base-uncased'):
        super(UnimodalEncoder, self).__init__()
        # Length Align (Learning Context Information)
        # self.len_align_v = nn.Linear(300, 50, bias=True)
        # self.len_align_a = nn.Linear(300, 50, bias=True)

        # All Encoders of Each Modality
        self.enc_t = UniPretrain(modality="T", pretrained=bert_pretrained, num_patches=50, proj_fea_dim=768)
        self.enc_v = UniPretrain(modality="V", num_patches=50, fea_size=20)
        self.enc_a = UniPretrain(modality="A", num_patches=50, fea_size=5)

    def forward(self, inputs_data_mask):
        # Encoder Part
        hidden_t, uni_T = self.enc_t(inputs_data_mask)
        hidden_v, uni_V = self.enc_v(inputs_data_mask)
        hidden_a, uni_A = self.enc_a(inputs_data_mask)

        return [hidden_t, hidden_v, hidden_a], []
