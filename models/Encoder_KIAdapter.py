import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
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
        x = x[:, 0, :]
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


class MixDomainAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class PositionEncodingTraining(nn.Module):
    """Construct the CLS token, position and patch embeddings.
    """
    def __init__(self, num_patches, fea_size, tf_hidden_dim, drop_out):
        super().__init__()
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


class TfEncoder(nn.Module):
    def __init__(self, fea_size, nhead, dim_feedforward, num_layers, dropout=0.2, activation='gelu'):
        super(TfEncoder, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionEncodingTraining(
            num_patches=50,
            fea_size=fea_size,
            tf_hidden_dim=dim_feedforward,
            drop_out=0.5
        )

        encoder_layers = TransformerEncoderLayer(dim_feedforward, nhead, dim_feedforward, dropout, activation=activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, has_mask=True, src_key_padding_mask=None):
        src = self.pos_encoder(src)

        src = src.transpose(0, 1)
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        output = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=src_key_padding_mask)

        return output.transpose(0, 1)


class UniEncoder(nn.Module):
    def __init__(self, m, pretrained, fea_size, nhead, dim_feedforward, num_layers, drop_out):
        super(UniEncoder, self).__init__()
        self.m = m

        if m in "VA":
            self.tfencoder = TfEncoder(
                fea_size=fea_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_layers=num_layers,
                dropout=drop_out,
                activation='gelu'
            )
            self.layernorm = nn.LayerNorm(dim_feedforward)
        else:
            self.model_config = BertConfig.from_pretrained(pretrained, output_hidden_states=True)
            self.tokenizer = BertTokenizer.from_pretrained(pretrained, do_lower_case=True)
            self.model = BertModel.from_pretrained(pretrained, config=self.model_config)
            # self.projector = FeatureProjector(fea_size, dim_feedforward)

    def forward(self, inputs, key_padding_mask):
        if self.m in "VA":
            x = self.tfencoder(inputs, has_mask=False, src_key_padding_mask=key_padding_mask)
            x = self.layernorm(x)
            x = torch.mean(x, dim=-2, keepdim=True)
            return x
        else:
            input_ids, input_mask, segment_ids = inputs[:, 0, :].long(), inputs[:, 1, :].float(), inputs[:, 2, :].long()    # 更换原始文本，使用tokenizer
            last_hidden_states = self.model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids
            )  # type: ignore # Models outputs are now tuples
            hidden_text = last_hidden_states[0]
            # hidden_text = self.projector(hidden_text)
            return hidden_text

    def get_tokenizer(self):
        return self.tokenizer


class UniPretrain(nn.Module):
    def __init__(self, modality, pretrained='./BERT/bert-base-chinese', fea_size=709, proj_fea_dim=1024, drop_out=0.3):
        super(UniPretrain, self).__init__()
        self.m = modality
        self.encoder = UniEncoder(
            m=modality,
            pretrained=pretrained,
            fea_size=fea_size,
            nhead=8,
            dim_feedforward=proj_fea_dim,
            num_layers=4,
            drop_out=drop_out
        )
        self.decoder = Classifier(
            input_size=proj_fea_dim,
            hidden_size=[int(proj_fea_dim / 2), int(proj_fea_dim / 4), int(proj_fea_dim / 8)],
            output_size=1,
            drop_out=drop_out
        )

    def forward(self, uni_fea):
        uni_fea, key_padding_mask = uni_fea[self.m], uni_fea["mask"][self.m]
        x = self.encoder(uni_fea, key_padding_mask)
        pred = self.decoder(x)
        return pred, x


class UnimodalAdapter(nn.Module):
    def __init__(self, opt, dataset, bert_pretrained='./BERT/bert-base-uncased'):
        super(UnimodalAdapter, self).__init__()
        # Input Dimension Align
        if dataset == 'mosi':
            self.dim_align_t = nn.Linear(768, 128)
            self.dim_align_v = nn.Linear(20, 128)
            self.dim_align_a = nn.Linear(5, 128)
        elif dataset == 'mosei':
            self.dim_align_t = nn.Linear(768, 128)
            self.dim_align_v = nn.Linear(35, 128)
            self.dim_align_a = nn.Linear(74, 128)
        elif dataset == 'sims':
            self.dim_align_t = nn.Linear(768, 128)
            self.dim_align_v = nn.Linear(709, 128)
            self.dim_align_a = nn.Linear(33, 128)
        else:
            assert False, "DatasetName must be mosi, mosei or sims."

        # Length Align (Learning Context Information)
        self.len_align_v = nn.Linear(300, 50, bias=True)
        self.len_align_a = nn.Linear(300, 50, bias=True)

        # All Encoders of Each Modality
        self.enc_t = UniPretrain(modality="T", pretrained=bert_pretrained)
        self.enc_v = UniPretrain(modality="V")
        self.enc_a = UniPretrain(modality="A")

    def forward(self, text, visual, acoustic):
        hidden_t = self.enc_t(text)

        # Dimension Align
        hidden_t = self.dim_align_t(hidden_t)
        visual = self.dim_align_v(visual)
        acoustic = self.dim_align_a(acoustic)

        # Length Align
        visual = self.len_align_v(visual.permute(0, 2, 1)).permute(0, 2, 1)
        acoustic = self.len_align_a(acoustic.permute(0, 2, 1)).permute(0, 2, 1)

        # Encoder Part
        hidden_v = self.enc_v(visual)
        hidden_a = self.enc_a(acoustic)

        return [hidden_t, hidden_v, hidden_a], []
